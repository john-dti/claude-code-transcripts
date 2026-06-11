"""Tests for the multi-session watch index: session scanning, per-thread repo
rendering, the index server routes, and the open/close lifecycle."""

import json
import threading
import time

import httpx
import pytest

import claude_code_transcripts as cct
from claude_code_transcripts import (
    scan_watch_sessions,
    new_watch_index_cache,
    render_content_block,
)

from test_watch import (
    _user_line,
    _ai_title_line,
    _assistant_line,
    _write_session,
    _serve_in_thread,
    _collect_sse,
    _data_for,
)


class TestScanWatchSessions:
    """scan_watch_sessions: flat, mtime-descending session rows for the index."""

    def test_returns_rows_sorted_newest_first(self, tmp_path):
        _write_session(tmp_path / "proj-a" / "old.jsonl", _user_line("old task"), 1000)
        _write_session(tmp_path / "proj-b" / "new.jsonl", _user_line("new task"), 2000)
        rows = scan_watch_sessions(tmp_path)
        assert [r["path"].name for r in rows] == ["new.jsonl", "old.jsonl"]

    def test_row_fields(self, tmp_path):
        data = _user_line("do the thing") + _ai_title_line("Doing the thing")
        _write_session(tmp_path / "D--projects-devjig" / "abc.jsonl", data, 1500)
        (row,) = scan_watch_sessions(tmp_path)
        assert row["path"] == tmp_path / "D--projects-devjig" / "abc.jsonl"
        assert row["project"] == "devjig"
        assert row["title"] == "Doing the thing"
        assert row["summary"] == "do the thing"
        assert row["mtime"] == 1500
        assert row["size"] > 0
        assert "branch" in row and "command" in row

    def test_excludes_agents_and_warmup_keeps_no_summary(self, tmp_path):
        _write_session(tmp_path / "p" / "agent-x.jsonl", _user_line("agent"), 1000)
        _write_session(tmp_path / "p" / "warm.jsonl", _user_line("warmup"), 1100)
        # A just-started session has no usable summary yet but is exactly what
        # a live watcher wants to see — it must stay in the index.
        _write_session(tmp_path / "p" / "fresh.jsonl", b'{"type":"x"}\n', 1200)
        rows = scan_watch_sessions(tmp_path)
        assert [r["path"].name for r in rows] == ["fresh.jsonl"]
        assert rows[0]["summary"] == "(no summary)"

    def test_missing_folder_returns_empty(self, tmp_path):
        assert scan_watch_sessions(tmp_path / "nope") == []

    def test_cache_skips_unchanged_files(self, tmp_path, monkeypatch):
        _write_session(tmp_path / "p" / "a.jsonl", _user_line("a"), 1000)
        _write_session(tmp_path / "p" / "b.jsonl", _user_line("b"), 2000)
        cache = new_watch_index_cache()
        scan_watch_sessions(tmp_path, cache=cache)

        calls = []
        real = cct.scan_session_metadata

        def counting(filepath, *args, **kwargs):
            calls.append(filepath)
            return real(filepath, *args, **kwargs)

        monkeypatch.setattr(cct, "scan_session_metadata", counting)
        rows = scan_watch_sessions(tmp_path, cache=cache)
        assert calls == []
        assert len(rows) == 2

    def test_cache_rescans_modified_file(self, tmp_path, monkeypatch):
        p = tmp_path / "p" / "a.jsonl"
        _write_session(p, _user_line("first"), 1000)
        cache = new_watch_index_cache()
        scan_watch_sessions(tmp_path, cache=cache)
        _write_session(p, _user_line("first") + _ai_title_line("Renamed"), 2000)

        calls = []
        real = cct.scan_session_metadata

        def counting(filepath, *args, **kwargs):
            calls.append(filepath)
            return real(filepath, *args, **kwargs)

        monkeypatch.setattr(cct, "scan_session_metadata", counting)
        (row,) = scan_watch_sessions(tmp_path, cache=cache)
        assert calls == [p]
        assert row["title"] == "Renamed"

    def test_deleted_file_drops_out(self, tmp_path):
        p = tmp_path / "p" / "a.jsonl"
        _write_session(p, _user_line("a"), 1000)
        cache = new_watch_index_cache()
        assert len(scan_watch_sessions(tmp_path, cache=cache)) == 1
        p.unlink()
        assert scan_watch_sessions(tmp_path, cache=cache) == []


class TestRenderRepoThreadLocal:
    """_current_github_repo: per-thread repo override with module-global fallback.

    Multiple sessions can point at different GitHub repos; each SSE handler
    thread must render commit links with its own session's repo without
    stomping a shared global."""

    COMMIT_BLOCK = {
        "type": "tool_result",
        "content": "[main abc1234] Add new feature\n 2 files changed",
        "is_error": False,
    }

    def test_falls_back_to_module_global(self):
        old = cct._github_repo
        cct._github_repo = "global/repo"
        try:
            html = str(render_content_block(self.COMMIT_BLOCK))
            assert "global/repo" in html
        finally:
            cct._github_repo = old

    def test_thread_override_wins_per_thread(self):
        old = cct._github_repo
        cct._github_repo = "global/repo"
        results = {}

        def render_with(repo, key):
            cct._set_render_repo(repo)
            results[key] = str(render_content_block(self.COMMIT_BLOCK))

        try:
            t1 = threading.Thread(target=render_with, args=("owner/alpha", "a"))
            t2 = threading.Thread(target=render_with, args=("owner/beta", "b"))
            t1.start(), t2.start()
            t1.join(timeout=5), t2.join(timeout=5)
            assert "owner/alpha" in results["a"] and "owner/beta" not in results["a"]
            assert "owner/beta" in results["b"] and "owner/alpha" not in results["b"]
            # This thread never set an override — global still applies here.
            assert "global/repo" in str(render_content_block(self.COMMIT_BLOCK))
        finally:
            cct._github_repo = old

    def test_thread_override_none_means_no_repo(self):
        old = cct._github_repo
        cct._github_repo = "global/repo"
        results = {}

        def render_without_repo():
            cct._set_render_repo(None)
            results["html"] = str(render_content_block(self.COMMIT_BLOCK))

        try:
            t = threading.Thread(target=render_without_repo)
            t.start()
            t.join(timeout=5)
            assert "global/repo" not in results["html"]
        finally:
            cct._github_repo = old
