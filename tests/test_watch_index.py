"""Tests for the multi-session watch index: session scanning, per-thread repo
rendering, the index server routes, and the open/close lifecycle."""

import json
import shutil
import threading
import time
from pathlib import Path

import httpx
import pytest

import claude_code_transcripts as cct
from claude_code_transcripts import (
    scan_watch_sessions,
    new_watch_index_cache,
    render_content_block,
    create_watch_server,
    create_live_server,
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

    def test_cache_prunes_deleted_files(self, tmp_path):
        # The cache contract, not just the row output: entries for vanished
        # files must not accumulate across rescans.
        p = tmp_path / "p" / "a.jsonl"
        _write_session(p, _user_line("a"), 1000)
        cache = new_watch_index_cache()
        scan_watch_sessions(tmp_path, cache=cache)
        assert len(cache) == 1
        p.unlink()
        scan_watch_sessions(tmp_path, cache=cache)
        assert cache == {}

    def test_cache_cleared_when_folder_vanishes(self, tmp_path):
        folder = tmp_path / "projects"
        _write_session(folder / "p" / "a.jsonl", _user_line("a"), 1000)
        cache = new_watch_index_cache()
        scan_watch_sessions(folder, cache=cache)
        assert len(cache) == 1
        shutil.rmtree(folder)
        assert scan_watch_sessions(folder, cache=cache) == []
        assert cache == {}


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


def _start_watch_server(folder, **kwargs):
    """create_watch_server + serve in a daemon thread; returns (server, port, thread)."""
    server = create_watch_server(folder, poll_interval=0.02, **kwargs)
    port = server.server_address[1]
    return server, port, _serve_in_thread(server)


def _get_sessions_json(port):
    r = httpx.get(f"http://127.0.0.1:{port}/api/sessions", timeout=5)
    assert r.status_code == 200
    return r.json()["sessions"]


def _poll_until(fn, timeout=5.0):
    """Re-evaluate `fn` until it returns truthy or the deadline passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = fn()
        if result:
            return result
        time.sleep(0.02)
    return fn()


class TestWatchServerRouting:
    """create_watch_server: index shell, sessions API, and per-session routes."""

    def test_root_serves_searchable_index(self, tmp_path):
        _write_session(tmp_path / "p" / "a.jsonl", _user_line("hello"), 1000)
        server, port, t = _start_watch_server(tmp_path)
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/", timeout=5)
            assert r.status_code == 200
            assert 'id="session-list"' in r.text
            assert 'id="index-search"' in r.text
            assert "api/sessions" in r.text  # the JS polls the sessions API
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_api_sessions_lists_rows(self, tmp_path):
        data = _user_line("fix the bug") + _ai_title_line("Fixing the bug")
        _write_session(tmp_path / "D--projects-devjig" / "abc.jsonl", data, 2000)
        _write_session(tmp_path / "p" / "old.jsonl", _user_line("older"), 1000)
        server, port, t = _start_watch_server(tmp_path)
        try:
            rows = _get_sessions_json(port)
            assert [r["id"] for r in rows] == ["abc", "old"]
            row = rows[0]
            assert row["title"] == "Fixing the bug"
            assert row["project"] == "devjig"
            assert row["url"] == "/session/abc/"
            assert row["watchers"] == 0
            assert row["mtime"] == 2000 and row["size"] > 0
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_session_url_without_slash_redirects(self, tmp_path):
        _write_session(tmp_path / "p" / "a.jsonl", _user_line("hello"), 1000)
        server, port, t = _start_watch_server(tmp_path)
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/session/a", timeout=5)
            assert r.status_code in (301, 308)
            assert r.headers["location"] == "/session/a/"
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_session_page_serves_live_shell_with_relative_events(self, tmp_path):
        data = _user_line("hello") + _ai_title_line("My session")
        _write_session(tmp_path / "p" / "a.jsonl", data, 1000)
        server, port, t = _start_watch_server(tmp_path)
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/session/a/", timeout=5)
            assert r.status_code == 200
            assert 'id="messages"' in r.text
            assert "My session" in r.text
            # Relative EventSource so the same shell works at any mount point.
            assert "EventSource('events')" in r.text
            assert "EventSource('/events')" not in r.text
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_unknown_session_404s(self, tmp_path):
        (tmp_path / "p").mkdir(parents=True)
        server, port, t = _start_watch_server(tmp_path)
        try:
            for url in ("/session/nope/", "/session/nope/events"):
                r = httpx.get(f"http://127.0.0.1:{port}{url}", timeout=5)
                assert r.status_code == 404
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_session_created_after_start_is_discovered(self, tmp_path):
        (tmp_path / "p").mkdir(parents=True)
        server, port, t = _start_watch_server(tmp_path)
        try:
            assert _get_sessions_json(port) == []
            _write_session(tmp_path / "p" / "late.jsonl", _user_line("new"), 3000)
            rows = _get_sessions_json(port)
            assert [r["id"] for r in rows] == ["late"]
            r = httpx.get(f"http://127.0.0.1:{port}/session/late/", timeout=5)
            assert r.status_code == 200
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_register_session_normalizes_relative_paths(self, tmp_path, monkeypatch):
        # `watch --session p/abc.jsonl` (relative) and the folder scan
        # (absolute) must resolve to ONE registry entry, or the index's
        # close button wouldn't close the CLI-opened watch.
        p = tmp_path / "p" / "abc.jsonl"
        _write_session(p, _user_line("hi"), 1000)
        server = create_watch_server(tmp_path, poll_interval=0.02)
        try:
            monkeypatch.chdir(tmp_path)
            rel = server.register_session(Path("p") / "abc.jsonl")
            rows = server.refresh_sessions()
            assert rel.id == "abc"
            assert [r["id"] for r in rows] == ["abc"]
            assert len(server.sessions) == 1
        finally:
            server.server_close()

    def test_legacy_single_session_server_keeps_old_routes(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_bytes(_user_line("hello"))
        server = create_live_server(p, poll_interval=0.02)
        port = server.server_address[1]
        t = _serve_in_thread(server)
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/", timeout=5)
            assert r.status_code == 200
            assert 'id="messages"' in r.text  # live shell, not the index
            r = httpx.get(f"http://127.0.0.1:{port}/api/sessions", timeout=5)
            assert r.status_code == 404
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)


class TestWatchServerMultiSession:
    """Concurrent SSE tails: each session streams its own file."""

    def test_two_sessions_stream_independently(self, tmp_path):
        _write_session(tmp_path / "p" / "aaa.jsonl", _user_line("alpha prompt"), 1000)
        _write_session(tmp_path / "p" / "bbb.jsonl", _user_line("beta prompt"), 2000)
        server, port, t = _start_watch_server(tmp_path)
        try:
            with (
                httpx.stream(
                    "GET", f"http://127.0.0.1:{port}/session/aaa/events", timeout=10
                ) as ra,
                httpx.stream(
                    "GET", f"http://127.0.0.1:{port}/session/bbb/events", timeout=10
                ) as rb,
            ):
                la, lb = ra.iter_lines(), rb.iter_lines()
                eva = _collect_sse(la, lambda evs: any(ev == "stats" for ev, _ in evs))
                evb = _collect_sse(lb, lambda evs: any(ev == "stats" for ev, _ in evs))
                html_a = "".join(d["html"] for d in _data_for(eva, "append"))
                html_b = "".join(d["html"] for d in _data_for(evb, "append"))
                assert "alpha prompt" in html_a and "beta prompt" not in html_a
                assert "beta prompt" in html_b and "alpha prompt" not in html_b
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)


class TestWatchersAndClose:
    """Watcher counts and closing a watched session from the index."""

    def test_watcher_count_tracks_connections(self, tmp_path):
        _write_session(tmp_path / "p" / "a.jsonl", _user_line("hello"), 1000)
        server, port, t = _start_watch_server(tmp_path)
        try:
            with httpx.stream(
                "GET", f"http://127.0.0.1:{port}/session/a/events", timeout=10
            ) as resp:
                lines = resp.iter_lines()
                _collect_sse(lines, lambda evs: any(ev == "stats" for ev, _ in evs))
                rows = _poll_until(
                    lambda: [r for r in _get_sessions_json(port) if r["watchers"] == 1]
                )
                assert rows and rows[0]["id"] == "a"
            rows = _poll_until(
                lambda: [r for r in _get_sessions_json(port) if r["watchers"] == 0]
            )
            assert rows and rows[0]["id"] == "a"
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_close_ends_stream_and_rearms(self, tmp_path):
        _write_session(tmp_path / "p" / "a.jsonl", _user_line("hello"), 1000)
        server, port, t = _start_watch_server(tmp_path)
        try:
            with httpx.stream(
                "GET", f"http://127.0.0.1:{port}/session/a/events", timeout=10
            ) as resp:
                lines = resp.iter_lines()
                _collect_sse(lines, lambda evs: any(ev == "stats" for ev, _ in evs))
                r = httpx.post(
                    f"http://127.0.0.1:{port}/api/sessions/a/close", timeout=5
                )
                assert r.status_code == 200
                events = _collect_sse(
                    lines, lambda evs: any(ev == "closed" for ev, _ in evs)
                )
                assert any(ev == "closed" for ev, _ in events)
                # Decrement happens on the close path itself — the client is
                # still connected here, so this can't pass via disconnect.
                rows = _poll_until(
                    lambda: [r for r in _get_sessions_json(port) if r["watchers"] == 0]
                )
                assert rows
                # Stream actually ends: a bounded drain sees no further
                # events. A regressed loop (e.g. missing break) re-emits
                # 'closed' forever and keepalives defeat the read timeout —
                # an unbounded list(lines) here would hang the suite.
                tail = _collect_sse(lines, lambda evs: False, timeout=1.5)
                assert tail == []
            # Close LATCHES: a bare reconnect (an orphaned tab's EventSource
            # auto-retry) is told 'closed' again instead of silently resuming.
            with httpx.stream(
                "GET", f"http://127.0.0.1:{port}/session/a/events", timeout=10
            ) as resp2:
                events = _collect_sse(
                    resp2.iter_lines(),
                    lambda evs: any(ev == "closed" for ev, _ in evs),
                )
                kinds = [ev for ev, _ in events]
                assert "closed" in kinds
                assert "append" not in kinds and "stats" not in kinds
            # Loading the session page is the deliberate re-open that re-arms.
            r = httpx.get(f"http://127.0.0.1:{port}/session/a/", timeout=5)
            assert r.status_code == 200
            with httpx.stream(
                "GET", f"http://127.0.0.1:{port}/session/a/events", timeout=10
            ) as resp3:
                events = _collect_sse(
                    resp3.iter_lines(),
                    lambda evs: any(ev == "stats" for ev, _ in evs),
                )
                assert any(ev == "reset" for ev, _ in events)
                assert any(ev == "append" for ev, _ in events)
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_close_unknown_session_404s(self, tmp_path):
        (tmp_path / "p").mkdir(parents=True)
        server, port, t = _start_watch_server(tmp_path)
        try:
            r = httpx.post(
                f"http://127.0.0.1:{port}/api/sessions/nope/close", timeout=5
            )
            assert r.status_code == 404
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_close_works_on_legacy_single_session_server(self, tmp_path):
        # Deliberate: the close API stays live in legacy mode (no index UI
        # links to it, but the registry exists and closing is harmless).
        p = tmp_path / "s.jsonl"
        p.write_bytes(_user_line("hello"))
        server = create_live_server(p, poll_interval=0.02)
        port = server.server_address[1]
        t = _serve_in_thread(server)
        try:
            with httpx.stream(
                "GET", f"http://127.0.0.1:{port}/events", timeout=10
            ) as resp:
                lines = resp.iter_lines()
                _collect_sse(lines, lambda evs: any(ev == "stats" for ev, _ in evs))
                r = httpx.post(
                    f"http://127.0.0.1:{port}/api/sessions/s/close", timeout=5
                )
                assert r.status_code == 200
                events = _collect_sse(
                    lines, lambda evs: any(ev == "closed" for ev, _ in evs)
                )
                assert any(ev == "closed" for ev, _ in events)
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)


class TestPercentEncodedSessionIds:
    """Session ids that need URL-encoding must round-trip through every route.

    Claude Code's UUID stems are URL-safe, but --session accepts arbitrary
    files; a stem with a space produced 404s on the page, the SSE stream,
    and the close endpoint (provenance: live probe with 'my session.jsonl'
    during the dti/watch-index adversarial review, 2026-06-11)."""

    def test_space_in_stem_is_watchable_and_closable(self, tmp_path):
        _write_session(tmp_path / "p" / "my session.jsonl", _user_line("hi"), 1000)
        server, port, t = _start_watch_server(tmp_path)
        try:
            (row,) = _get_sessions_json(port)
            assert row["id"] == "my session"
            assert row["url"] == "/session/my%20session/"
            r = httpx.get(f"http://127.0.0.1:{port}/session/my%20session/", timeout=5)
            assert r.status_code == 200
            r = httpx.post(
                f"http://127.0.0.1:{port}/api/sessions/my%20session/close", timeout=5
            )
            assert r.status_code == 200
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)


class TestRequestValidation:
    """Host/Origin checks: a loopback dev server still needs DNS-rebinding
    and cross-site POST protection — /api/sessions exposes transcript titles."""

    def test_bad_host_header_is_forbidden(self, tmp_path):
        _write_session(tmp_path / "p" / "a.jsonl", _user_line("hello"), 1000)
        server, port, t = _start_watch_server(tmp_path)
        try:
            for url in ("/", "/api/sessions", "/session/a/"):
                r = httpx.get(
                    f"http://127.0.0.1:{port}{url}",
                    headers={"Host": "evil.example.com"},
                    timeout=5,
                )
                assert r.status_code == 403, url
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_cross_origin_close_post_is_forbidden(self, tmp_path):
        _write_session(tmp_path / "p" / "a.jsonl", _user_line("hello"), 1000)
        server, port, t = _start_watch_server(tmp_path)
        try:
            _get_sessions_json(port)  # populate the registry, as the index does
            r = httpx.post(
                f"http://127.0.0.1:{port}/api/sessions/a/close",
                headers={"Origin": "http://evil.example.com"},
                timeout=5,
            )
            assert r.status_code == 403
            # Same-origin POST (what INDEX_JS sends) still works.
            r = httpx.post(
                f"http://127.0.0.1:{port}/api/sessions/a/close",
                headers={"Origin": f"http://127.0.0.1:{port}"},
                timeout=5,
            )
            assert r.status_code == 200
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)


class TestRegistryCollisions:
    """Two same-stem files from different projects must both stay reachable."""

    def test_same_stem_sessions_get_distinct_ids(self, tmp_path):
        _write_session(tmp_path / "proj-a" / "abc.jsonl", _user_line("alpha"), 2000)
        _write_session(tmp_path / "proj-b" / "abc.jsonl", _user_line("beta"), 1000)
        server, port, t = _start_watch_server(tmp_path)
        try:
            rows = _get_sessions_json(port)
            ids = [r["id"] for r in rows]
            assert len(set(ids)) == 2
            assert ids[0] == "abc"  # newest registers first, keeps the stem
            assert ids[1].startswith("abc~")
            # Each id serves its own file (shell is pre-titled per session).
            for sid, marker in [(ids[0], "alpha"), (ids[1], "beta")]:
                r = httpx.get(f"http://127.0.0.1:{port}/session/{sid}/", timeout=5)
                assert r.status_code == 200
                assert marker in r.text
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)
