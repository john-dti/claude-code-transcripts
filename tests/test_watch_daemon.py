"""Tests for the watch daemon: identity/registration/shutdown endpoints,
the on-disk state file, the background-by-default CLI, and --stop."""

import json
import os
import threading
import time
from pathlib import Path

import httpx
import pytest

import claude_code_transcripts as cct
from claude_code_transcripts import create_live_server, create_watch_server

from test_watch import _serve_in_thread, _user_line, _write_session
from test_watch_index import _poll_until, _start_watch_server


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    """Point the daemon state dir at a temp folder for the whole test."""
    d = tmp_path / "state"
    monkeypatch.setenv("CLAUDE_CODE_TRANSCRIPTS_STATE_DIR", str(d))
    return d


class TestWatchStateFile:
    """The on-disk record of a running watch server (pid/port/urls)."""

    def test_write_then_read_roundtrip(self, tmp_path, state_dir):
        cct._write_watch_state(
            tmp_path, pid=123, port=456, open_url="http://127.0.0.1:456/"
        )
        state = cct._read_watch_state(tmp_path)
        assert state["pid"] == 123
        assert state["port"] == 456
        assert state["open_url"] == "http://127.0.0.1:456/"
        assert Path(state["source"]) == tmp_path.resolve()

    def test_read_missing_returns_none(self, tmp_path, state_dir):
        assert cct._read_watch_state(tmp_path) is None

    def test_read_corrupt_returns_none(self, tmp_path, state_dir):
        path = cct._watch_state_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json{", encoding="utf-8")
        assert cct._read_watch_state(tmp_path) is None

    def test_clear_removes_state(self, tmp_path, state_dir):
        cct._write_watch_state(tmp_path, pid=1, port=2, open_url="u")
        cct._clear_watch_state(tmp_path)
        assert cct._read_watch_state(tmp_path) is None
        cct._clear_watch_state(tmp_path)  # idempotent when already gone

    def test_state_paths_scoped_per_source(self, tmp_path, state_dir):
        a = cct._watch_state_path(tmp_path / "a")
        b = cct._watch_state_path(tmp_path / "b")
        assert a != b
        assert a.parent == b.parent == state_dir

    def test_state_path_stable_across_spellings(self, tmp_path, state_dir, monkeypatch):
        monkeypatch.chdir(tmp_path)
        absolute = cct._watch_state_path(tmp_path / "proj")
        relative = cct._watch_state_path(Path("proj"))
        assert absolute == relative

    def test_log_path_sits_beside_state(self, tmp_path, state_dir):
        state = cct._watch_state_path(tmp_path)
        log = cct._watch_log_path(tmp_path)
        assert log.parent == state.parent
        assert log.stem == state.stem
        assert log.suffix == ".log"


class TestProbeWatchServer:
    """_probe_watch_server: trust a recorded server only if it answers
    /api/watch-info as OUR app serving the SAME projects folder."""

    def test_healthy_matching_server_returns_index_url(self, tmp_path, state_dir):
        server, port, t = _start_watch_server(tmp_path)
        try:
            url = cct._probe_watch_server({"port": port}, tmp_path)
            assert url == f"http://127.0.0.1:{port}/"
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_dead_port_returns_none(self, tmp_path, state_dir):
        server, port, t = _start_watch_server(tmp_path)
        server.shutdown()
        server.server_close()
        t.join(timeout=5)
        assert cct._probe_watch_server({"port": port}, tmp_path) is None

    def test_server_for_other_source_returns_none(self, tmp_path, state_dir):
        other = tmp_path / "other"
        other.mkdir()
        server, port, t = _start_watch_server(other)
        try:
            assert cct._probe_watch_server({"port": port}, tmp_path) is None
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_none_or_incomplete_state_returns_none(self, tmp_path, state_dir):
        assert cct._probe_watch_server(None, tmp_path) is None
        assert cct._probe_watch_server({}, tmp_path) is None


class TestWatchInfoEndpoint:
    """GET /api/watch-info identifies the server so relaunches can trust it."""

    def test_reports_app_pid_and_source(self, tmp_path):
        server, port, t = _start_watch_server(tmp_path)
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/api/watch-info", timeout=5)
            assert r.status_code == 200
            info = r.json()
            assert info["app"] == "claude-code-transcripts"
            assert info["mode"] == "watch"
            assert info["pid"] == os.getpid()
            assert Path(info["source"]) == tmp_path
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_404_on_legacy_single_session_server(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_bytes(_user_line("hi"))
        server = create_live_server(p, poll_interval=0.02)
        port = server.server_address[1]
        t = _serve_in_thread(server)
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/api/watch-info", timeout=5)
            assert r.status_code == 404
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)


class TestOpenSessionEndpoint:
    """POST /api/sessions/open registers a session file on a running server,
    so `watch --session X` can target an already-running daemon."""

    def test_registers_and_returns_session_url(self, tmp_path):
        p = tmp_path / "proj" / "abc.jsonl"
        _write_session(p, _user_line("hello"), 2000)
        server, port, t = _start_watch_server(tmp_path)
        try:
            r = httpx.post(
                f"http://127.0.0.1:{port}/api/sessions/open",
                json={"path": str(p)},
                timeout=5,
            )
            assert r.status_code == 200
            assert r.json()["url"] == "/session/abc/"
            # The returned URL must actually serve the live shell.
            shell = httpx.get(f"http://127.0.0.1:{port}/session/abc/", timeout=5)
            assert shell.status_code == 200
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_missing_file_is_400(self, tmp_path):
        server, port, t = _start_watch_server(tmp_path)
        try:
            r = httpx.post(
                f"http://127.0.0.1:{port}/api/sessions/open",
                json={"path": str(tmp_path / "nope.jsonl")},
                timeout=5,
            )
            assert r.status_code == 400
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_bad_json_body_is_400(self, tmp_path):
        server, port, t = _start_watch_server(tmp_path)
        try:
            r = httpx.post(
                f"http://127.0.0.1:{port}/api/sessions/open",
                content=b"not json",
                timeout=5,
            )
            assert r.status_code == 400
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_cross_origin_post_is_forbidden(self, tmp_path):
        p = tmp_path / "proj" / "abc.jsonl"
        _write_session(p, _user_line("hello"), 2000)
        server, port, t = _start_watch_server(tmp_path)
        try:
            r = httpx.post(
                f"http://127.0.0.1:{port}/api/sessions/open",
                json={"path": str(p)},
                headers={"Origin": "https://evil.example"},
                timeout=5,
            )
            assert r.status_code == 403
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)


class TestShutdownEndpoint:
    """POST /api/shutdown stops serve_forever so --stop can end a daemon."""

    def test_shutdown_stops_serve_loop(self, tmp_path):
        server = create_watch_server(tmp_path, poll_interval=0.02)
        port = server.server_address[1]
        t = _serve_in_thread(server)
        try:
            r = httpx.post(f"http://127.0.0.1:{port}/api/shutdown", timeout=5)
            assert r.status_code == 200
            assert r.json() == {"stopping": True}
            t.join(timeout=5)
            assert not t.is_alive()
        finally:
            if t.is_alive():  # only on failure — normal path already stopped
                server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_cross_origin_shutdown_is_forbidden(self, tmp_path):
        server, port, t = _start_watch_server(tmp_path)
        try:
            r = httpx.post(
                f"http://127.0.0.1:{port}/api/shutdown",
                headers={"Origin": "https://evil.example"},
                timeout=5,
            )
            assert r.status_code == 403
            assert t.is_alive()
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)
