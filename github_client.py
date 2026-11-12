"""GitHub REST API client for public repository issues and comments."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_BASE = "https://api.github.com"


class GitHubAPIError(RuntimeError):
    """Raised when GitHub returns an unsuccessful API response."""


class GitHubClient:
    def __init__(self, token: str | None = None, timeout: int = 25) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "github-issue-sentiment-monitor/1.0",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token.strip()}"

        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.7,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)

    @staticmethod
    def _iso(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        response = self.session.get(url, params=params, timeout=self.timeout)
        if response.status_code >= 400:
            message = f"GitHub API вернул HTTP {response.status_code}."
            try:
                payload = response.json()
                if payload.get("message"):
                    message += f" {payload['message']}"
            except ValueError:
                pass
            if response.status_code == 403:
                remaining = response.headers.get("X-RateLimit-Remaining")
                reset = response.headers.get("X-RateLimit-Reset")
                if remaining == "0" and reset:
                    reset_dt = datetime.fromtimestamp(int(reset), tz=timezone.utc)
                    message += f" Лимит API исчерпан; сброс примерно в {reset_dt:%H:%M:%S UTC}."
            if response.status_code == 404:
                message += " Проверьте имя репозитория и его публичную доступность."
            raise GitHubAPIError(message)
        return response

    def _paginate(
        self,
        url: str,
        params: dict[str, Any],
        max_pages: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        next_url: str | None = url
        next_params: dict[str, Any] | None = params

        for _ in range(max_pages):
            if not next_url:
                break
            response = self._get(next_url, params=next_params)
            payload = response.json()
            if not isinstance(payload, list):
                raise GitHubAPIError("GitHub API вернул неожиданный формат ответа.")
            results.extend(item for item in payload if isinstance(item, dict))
            next_link = response.links.get("next", {}).get("url")
            next_url = next_link
            next_params = None  # The Link URL already contains its query parameters.
            if not next_link:
                break
        return results

    def list_issues(
        self,
        repo: str,
        since: datetime,
        max_pages: int = 2,
        start_page: int = 1,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        """Return issues, pages fetched, and whether the API has another page."""
        owner_repo = repo.strip().strip("/")
        url = f"{API_BASE}/repos/{owner_repo}/issues"
        params: dict[str, Any] = {
            "state": "all",
            "sort": "updated",
            "direction": "asc",
            "since": self._iso(since),
            "per_page": 100,
            "page": start_page,
        }
        results: list[dict[str, Any]] = []
        next_url: str | None = url
        next_params: dict[str, Any] | None = params
        pages_fetched = 0
        has_more = False
        while next_url and pages_fetched < max_pages:
            response = self._get(next_url, params=next_params)
            payload = response.json()
            if not isinstance(payload, list):
                raise GitHubAPIError("GitHub API вернул неожиданный формат списка Issues.")
            results.extend(item for item in payload if isinstance(item, dict))
            pages_fetched += 1
            next_link = response.links.get("next", {}).get("url")
            has_more = bool(next_link)
            next_url = next_link
            next_params = None
        return results, pages_fetched, has_more

    def list_issue_comments(
        self,
        repo: str,
        issue_number: int,
        since: datetime | None = None,
        max_pages: int = 3,
    ) -> list[dict[str, Any]]:
        owner_repo = repo.strip().strip("/")
        url = f"{API_BASE}/repos/{owner_repo}/issues/{issue_number}/comments"
        params: dict[str, Any] = {"per_page": 100}
        if since is not None:
            params["since"] = self._iso(since)
        return self._paginate(url, params, max_pages=max_pages)


def parse_github_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def sync_repository(
    client: GitHubClient,
    database: Any,
    repo: str,
    lookback_days: int = 7,
    max_issue_pages: int = 2,
    max_comment_pages: int = 3,
) -> dict[str, int]:
    """Fetch changed issues and their changed comments, then store raw text."""
    now = datetime.now(timezone.utc)
    repo_state = database.get_repo_state(repo)
    previous_sync = repo_state["last_sync_at"] if repo_state else None
    if repo_state and repo_state["pending_since"]:
        # Continue a paginated initial/backfill query instead of skipping older pages.
        since = repo_state["pending_since"]
        start_page = repo_state["next_page"]
    elif previous_sync:
        # A small overlap protects against records sharing the same update timestamp.
        since = previous_sync - timedelta(minutes=2)
        start_page = 1
    else:
        since = now - timedelta(days=lookback_days)
        start_page = 1

    issues, pages_fetched, has_more_pages = client.list_issues(
        repo, since=since, max_pages=max_issue_pages, start_page=start_page
    )
    raw_messages: list[dict[str, Any]] = []
    comment_updates = 0
    issue_count = 0

    for issue in issues:
        # The Issues endpoint also returns pull requests.
        if "pull_request" in issue:
            continue

        issue_number = int(issue["number"])
        issue_count += 1
        title = (issue.get("title") or "").strip()
        body = (issue.get("body") or "").strip()
        issue_text = f"{title}\n\n{body}".strip()

        if issue_text:
            raw_messages.append(
                {
                    "message_id": f"issue:{repo}#{issue_number}",
                    "repo": repo,
                    "kind": "issue",
                    "issue_number": issue_number,
                    "title": title,
                    "text": issue_text,
                    "username": (issue.get("user") or {}).get("login", "unknown"),
                    "created_at": issue.get("created_at"),
                    "updated_at": issue.get("updated_at"),
                    "html_url": issue.get("html_url", ""),
                    "labels": [label.get("name", "") for label in issue.get("labels", [])],
                }
            )

        if int(issue.get("comments") or 0) <= 0:
            continue

        previous_comment_cursor = database.get_comment_cursor(repo, issue_number)
        comment_since = previous_comment_cursor - timedelta(seconds=2) if previous_comment_cursor else None
        comments = client.list_issue_comments(
            repo,
            issue_number,
            since=comment_since,
            max_pages=max_comment_pages,
        )

        latest_comment_update = previous_comment_cursor
        for comment in comments:
            comment_body = (comment.get("body") or "").strip()
            if not comment_body:
                continue
            raw_messages.append(
                {
                    "message_id": f"comment:{comment['id']}",
                    "repo": repo,
                    "kind": "comment",
                    "issue_number": issue_number,
                    "title": f"Комментарий к #{issue_number}: {title}",
                    "text": comment_body,
                    "username": (comment.get("user") or {}).get("login", "unknown"),
                    "created_at": comment.get("created_at"),
                    "updated_at": comment.get("updated_at") or comment.get("created_at"),
                    "html_url": comment.get("html_url", ""),
                    "labels": [],
                }
            )
            parsed_update = parse_github_timestamp(comment.get("updated_at") or comment.get("created_at"))
            if parsed_update and (latest_comment_update is None or parsed_update > latest_comment_update):
                latest_comment_update = parsed_update

        if latest_comment_update:
            database.set_comment_cursor(repo, issue_number, latest_comment_update)
        comment_updates += len(comments)

    changed = database.upsert_raw_messages(raw_messages)
    if has_more_pages:
        # The next polling cycle resumes at the next page for the same since value.
        database.set_repo_cursor(
            repo,
            previous_sync or since,
            pending_since=since,
            next_page=start_page + pages_fetched,
        )
    else:
        database.set_repo_cursor(repo, now, pending_since=None, next_page=1)
    return {
        "issues_seen": issue_count,
        "comments_seen": comment_updates,
        "messages_changed": len(changed),
        "pages_fetched": pages_fetched,
        "more_pages_pending": int(has_more_pages),
    }
