from datetime import datetime, timedelta, timezone

import pandas as pd

from analytics import detect_negative_spikes, prepare_frame


def make_frame() -> pd.DataFrame:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rows = []
    # A stable neutral baseline.
    for i in range(24):
        stamp = now - timedelta(minutes=15 * (i + 1))
        rows.append({"message_id": f"n{i}", "created_at": stamp, "sentiment": "neutral", "repo": "demo/repo", "kind": "issue", "text": "Question", "title": "Question"})
    # A short negative burst in the current 15-minute bucket.
    for i in range(4):
        rows.append({"message_id": f"b{i}", "created_at": now + timedelta(minutes=2 * i), "sentiment": "negative", "repo": "demo/repo", "kind": "comment", "text": "Broken and failing", "title": "Failure"})
    return pd.DataFrame(rows)


def test_prepare_frame_adds_display_columns():
    frame = prepare_frame(make_frame())
    assert "sentiment_ru" in frame.columns
    assert "text_preview" in frame.columns


def test_negative_spike_is_detected():
    frame = prepare_frame(make_frame())
    spikes = detect_negative_spikes(frame)
    assert not spikes.empty
    assert spikes["is_spike"].any()


def test_demo_database_is_idempotent(tmp_path):
    from database import Database
    from demo_data import build_demo_messages

    database = Database(tmp_path / "test.sqlite")
    first = database.insert_demo_messages(build_demo_messages())
    second = database.insert_demo_messages(build_demo_messages())
    assert first == 22
    assert second == 0
    assert len(database.load_messages()) == 22


def test_issue_sync_continues_on_next_page(tmp_path):
    from datetime import datetime, timezone

    from database import Database
    from github_client import sync_repository

    class FakeClient:
        def list_issues(self, repo, since, max_pages=2, start_page=1):
            issue = {
                "number": start_page,
                "title": f"Issue {start_page}",
                "body": "Example body",
                "user": {"login": "tester"},
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": f"2026-01-01T00:0{start_page}:00Z",
                "html_url": "https://github.com/example/repo/issues/1",
                "labels": [],
                "comments": 0,
            }
            return [issue], 1, start_page == 1

        def list_issue_comments(self, repo, issue_number, since=None, max_pages=3):
            return []

    database = Database(tmp_path / "sync.sqlite")
    fake = FakeClient()
    first = sync_repository(fake, database, "example/repo", lookback_days=14, max_issue_pages=1)
    state_after_first = database.get_repo_state("example/repo")
    second = sync_repository(fake, database, "example/repo", lookback_days=14, max_issue_pages=1)
    state_after_second = database.get_repo_state("example/repo")

    assert first["more_pages_pending"] == 1
    assert state_after_first["next_page"] == 2
    assert second["more_pages_pending"] == 0
    assert state_after_second["pending_since"] is None
    assert len(database.load_messages()) == 2
