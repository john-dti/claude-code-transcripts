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
