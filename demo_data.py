"""Small synthetic dataset for previewing the dashboard without GitHub credentials."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


REPOSITORIES = ["pallets/flask", "streamlit/streamlit", "psf/requests"]


def build_demo_messages() -> list[dict]:
    now = datetime.now(timezone.utc)
    examples = [
        ("issue", "negative", "Application crashes when starting", "The application crashes on startup after the latest update. Please fix this regression.", "alice-dev"),
        ("comment", "negative", "Comment: startup crash", "I can reproduce this on Linux. The error makes the package unusable for our team.", "bob-code"),
        ("comment", "negative", "Comment: startup crash", "Same issue here. This is extremely frustrating and the workaround does not work.", "charlie"),
        ("issue", "neutral", "Question about configuration", "Is there a documented way to configure the application behind a reverse proxy?", "dev-user"),
        ("comment", "positive", "Comment: configuration", "The example in the documentation worked perfectly. Thank you!", "maintainer"),
        ("issue", "negative", "Memory usage increases over time", "Memory consumption keeps growing during long-running sessions and eventually the process stops responding.", "ops-team"),
        ("comment", "negative", "Comment: memory usage", "We are seeing the same memory leak in production. This needs attention.", "qa-engineer"),
        ("issue", "positive", "Add support for new parameter", "It would be great to support this option. The proposed implementation is simple and backwards compatible.", "contributor"),
        ("comment", "positive", "Comment: new parameter", "Thanks for the quick review. The new option works as expected.", "alice-dev"),
        ("issue", "neutral", "Documentation wording", "The installation section mentions an environment variable but does not show its default value.", "reader"),
        ("comment", "negative", "Comment: documentation", "The instructions are outdated and the command shown in the guide fails.", "new-user"),
        ("issue", "negative", "Regression after dependency update", "After updating the dependency, requests fail with a timeout. Several users have reported this today.", "net-user"),
        ("comment", "negative", "Comment: dependency update", "Confirmed. Rolling back the dependency is the only way to restore service.", "maintainer-2"),
        ("comment", "neutral", "Comment: dependency update", "Could you share the exact version and traceback?", "triager"),
        ("issue", "positive", "Feature request: export report", "An export option would make the workflow much easier. I can help test a pull request.", "community"),
        ("comment", "neutral", "Comment: export report", "There is an open discussion about the export format in the linked issue.", "maintainer"),
        ("issue", "negative", "Login occasionally fails", "Users receive an internal server error when logging in. The failure rate increased after deployment.", "service-owner"),
        ("comment", "negative", "Comment: login", "This is blocking our release. Please investigate as soon as possible.", "release-manager"),
        ("comment", "positive", "Comment: login", "The patch resolved the issue in our test environment. Great work.", "qa-engineer"),
        ("issue", "neutral", "Clarify API response format", "Can the API response schema be documented, including optional fields?", "api-consumer"),
        ("issue", "positive", "Thanks for the latest release", "The latest release fixed our previous issue and performance is noticeably better.", "happy-user"),
        ("comment", "negative", "Comment: latest release", "Unfortunately the release also introduced a new error for Windows users.", "windows-dev"),
    ]
    rows: list[dict] = []
    for index, (kind, sentiment, title, text, user) in enumerate(examples):
        # Create a compact burst of negative comments in the last hour to demonstrate spike detection.
        if index in {1, 2, 6, 7, 12, 13, 17, 21}:
            minutes_ago = 7 + (index % 4) * 2
        else:
            minutes_ago = (index + 1) * 85
        created = now - timedelta(minutes=minutes_ago)
        repo = REPOSITORIES[index % len(REPOSITORIES)]
        number = 100 + index
        rows.append(
            {
                "message_id": f"demo:{index}",
                "repo": repo,
                "kind": kind,
                "issue_number": number,
                "title": title,
                "text": text,
                "username": user,
                "created_at": created.isoformat(),
                "updated_at": created.isoformat(),
                "html_url": f"https://github.com/{repo}/issues/{number}",
                "labels": ["bug"] if sentiment == "negative" else [],
                "sentiment": sentiment,
                "score": 0.86 if sentiment == "negative" else (0.91 if sentiment == "positive" else 0.72),
            }
        )
    return rows
