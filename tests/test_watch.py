"""Tests for the live `watch` command: JSONL tailing, SSE framing, and the server."""

import json
import os
import threading
import time
from pathlib import Path

import httpx
from click.testing import CliRunner

from claude_code_transcripts import (
    _normalize_jsonl_obj,
    _parse_jsonl_file,
    read_new_loglines,
    format_sse_event,
    render_logline,
    index_prompt,
    new_live_stats,
    accumulate_live_stats,
    live_stats_payload,
    resolve_active_session,
    create_live_server,
    compute_usage_totals,
    last_assistant_snippet,
    prompt_preview,
    cli,
)


def _user_line(text, ts="T"):
    """One JSONL user-message line (newline-terminated) as bytes."""
    obj = {
        "type": "user",
        "timestamp": ts,
        "message": {"role": "user", "content": text},
    }
    return (json.dumps(obj) + "\n").encode("utf-8")


def _ai_title_line(title):
    """One JSONL ai-title line — Claude Code's evolving session name.

    Shape verified against real ~/.claude/projects files (Claude Code v2.1.x,
    2026-06-10): {"type":"ai-title","aiTitle":"...","sessionId":"..."}
    """
    obj = {"type": "ai-title", "aiTitle": title, "sessionId": "s"}
    return (json.dumps(obj) + "\n").encode("utf-8")


def _write_session(path, data, mtime):
    """Write a session file under `path` and stamp its mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.utime(path, (mtime, mtime))


def _collect_sse(lines, predicate, timeout=5.0):
    """Parse SSE events from a shared line iterator until `predicate(events)` is
    True or the wall-clock deadline passes. Returns the (event, data) list.

    Takes the iterator (not the response) so multiple phases can resume the same
    stream — httpx forbids calling iter_lines() twice. Never sleeps-then-asserts:
    the server's periodic `: keepalive` keeps the iterator returning so the
    deadline is enforced even when idle.
    """
    deadline = time.monotonic() + timeout
    events = []
    cur_event = None
    cur_data = ""
    for line in lines:
        if line.startswith("event:"):
            cur_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            cur_data += line[len("data:") :].strip()
        elif line == "":
            if cur_event is not None:
                events.append((cur_event, cur_data))
                cur_event, cur_data = None, ""
                if predicate(events):
                    return events
        if time.monotonic() > deadline:
            return events
    return events


def _data_for(events, event_name):
    """All decoded JSON payloads for a given event type, in order."""
    return [json.loads(d) for ev, d in events if ev == event_name]


def _serve_in_thread(server):
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t


class TestNormalizeJsonlObj:
    """The per-line normalizer shared by _parse_jsonl_file and the tail reader."""

    def test_user_entry(self):
        obj = {
            "type": "user",
            "timestamp": "T1",
            "message": {"role": "user", "content": "hi"},
            "uuid": "u1",
        }
        assert _normalize_jsonl_obj(obj) == {
            "type": "user",
            "timestamp": "T1",
            "message": {"role": "user", "content": "hi"},
        }

    def test_assistant_entry(self):
        obj = {
            "type": "assistant",
            "timestamp": "T2",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
            },
        }
        assert _normalize_jsonl_obj(obj) == {
            "type": "assistant",
            "timestamp": "T2",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
            },
        }

    def test_compact_summary_preserved(self):
        obj = {
            "type": "user",
            "timestamp": "T3",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "cont"},
        }
        entry = _normalize_jsonl_obj(obj)
        assert entry["isCompactSummary"] is True

    def test_missing_timestamp_defaults_empty(self):
        obj = {"type": "user", "message": {"role": "user", "content": "hi"}}
        assert _normalize_jsonl_obj(obj)["timestamp"] == ""

    def test_summary_type_skipped(self):
        assert _normalize_jsonl_obj({"type": "summary", "summary": "S"}) is None

    def test_other_type_skipped(self):
        assert _normalize_jsonl_obj({"type": "file-history-snapshot", "foo": 1}) is None

    def test_ai_title_dropped_by_default(self):
        """The static parse path never sees meta entries — contract unchanged."""
        obj = {"type": "ai-title", "aiTitle": "Name", "sessionId": "x"}
        assert _normalize_jsonl_obj(obj) is None

    def test_ai_title_with_include_meta(self):
        obj = {"type": "ai-title", "aiTitle": "Name", "sessionId": "x"}
        assert _normalize_jsonl_obj(obj, include_meta=True) == {
            "type": "ai-title",
            "title": "Name",
        }

    def test_empty_ai_title_dropped_with_include_meta(self):
        assert (
            _normalize_jsonl_obj({"type": "ai-title", "aiTitle": ""}, include_meta=True)
            is None
        )

    def test_system_dropped_even_with_include_meta(self):
        obj = {"type": "system", "subtype": "turn_duration", "timestamp": "T"}
        assert _normalize_jsonl_obj(obj, include_meta=True) is None


class TestParseJsonlFileCharacterization:
    """Pin _parse_jsonl_file output across every branch (refactor safety net)."""

    def test_all_branches(self, tmp_path):
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text(
            '{"type":"summary","summary":"S","leafUuid":"x"}\n'
            '{"type":"user","timestamp":"T1","message":{"role":"user","content":"hi"},"uuid":"u1"}\n'
            '{"type":"assistant","timestamp":"T2","message":{"role":"assistant","content":[{"type":"text","text":"hello"}]}}\n'
            '{"type":"user","timestamp":"T3","isCompactSummary":true,"message":{"role":"user","content":"cont"}}\n'
            '{"type":"file-history-snapshot","foo":1}\n'
            "\n"
            "{not valid json\n",
            encoding="utf-8",
        )

        assert _parse_jsonl_file(jsonl) == {
            "loglines": [
                {
                    "type": "user",
                    "timestamp": "T1",
                    "message": {"role": "user", "content": "hi"},
                },
                {
                    "type": "assistant",
                    "timestamp": "T2",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "hello"}],
                    },
                },
                {
                    "type": "user",
                    "timestamp": "T3",
                    "message": {"role": "user", "content": "cont"},
                    "isCompactSummary": True,
                },
            ]
        }


class TestReadNewLoglines:
    """Incremental byte-offset tail reader."""

    def test_reads_all_from_zero(self, tmp_path):
        p = tmp_path / "s.jsonl"
        data = _user_line("one", "T1") + _user_line("two", "T2")
        p.write_bytes(data)

        loglines, offset = read_new_loglines(p, 0)

        assert [e["message"]["content"] for e in loglines] == ["one", "two"]
        assert offset == len(data)

    def test_append_returns_only_new(self, tmp_path):
        p = tmp_path / "s.jsonl"
        first = _user_line("one", "T1")
        p.write_bytes(first)
        _, offset = read_new_loglines(p, 0)

        with open(p, "ab") as f:
            f.write(_user_line("two", "T2"))

        loglines, new_offset = read_new_loglines(p, offset)

        assert [e["message"]["content"] for e in loglines] == ["two"]
        assert new_offset == p.stat().st_size

    def test_partial_line_not_consumed_then_completed(self, tmp_path):
        p = tmp_path / "s.jsonl"
        complete = _user_line("one", "T1")
        partial = (
            b'{"type":"user","timestamp":"T2","message":{"role":"user","content":"tw'
        )
        p.write_bytes(complete + partial)

        loglines, offset = read_new_loglines(p, 0)

        # Only the complete line is consumed; offset sits before the partial line.
        assert [e["message"]["content"] for e in loglines] == ["one"]
        assert offset == len(complete)

        # Finish the partial line; the next read picks it up.
        with open(p, "ab") as f:
            f.write(b'o"}}\n')

        loglines2, offset2 = read_new_loglines(p, offset)
        assert [e["message"]["content"] for e in loglines2] == ["two"]
        assert offset2 == p.stat().st_size

    def test_no_newline_yet_consumes_nothing(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_bytes(b'{"type":"user"')  # no trailing newline

        loglines, offset = read_new_loglines(p, 0)

        assert loglines == []
        assert offset == 0

    def test_non_message_line_skipped_but_offset_advances(self, tmp_path):
        p = tmp_path / "s.jsonl"
        summary = b'{"type":"summary","summary":"S"}\n'
        data = summary + _user_line("one", "T1")
        p.write_bytes(data)

        loglines, offset = read_new_loglines(p, 0)

        assert [e["message"]["content"] for e in loglines] == ["one"]
        assert offset == len(data)  # advanced past the skipped summary too

    def test_malformed_line_skipped(self, tmp_path):
        p = tmp_path / "s.jsonl"
        data = b"{not valid json\n" + _user_line("one", "T1")
        p.write_bytes(data)

        loglines, offset = read_new_loglines(p, 0)

        assert [e["message"]["content"] for e in loglines] == ["one"]
        assert offset == len(data)

    def test_blank_line_skipped(self, tmp_path):
        p = tmp_path / "s.jsonl"
        data = b"\n" + _user_line("one", "T1")
        p.write_bytes(data)

        loglines, offset = read_new_loglines(p, 0)

        assert [e["message"]["content"] for e in loglines] == ["one"]
        assert offset == len(data)

    def test_crlf_line_parses(self, tmp_path):
        p = tmp_path / "s.jsonl"
        line = _user_line("one", "T1").rstrip(b"\n") + b"\r\n"
        p.write_bytes(line)

        loglines, offset = read_new_loglines(p, 0)

        assert [e["message"]["content"] for e in loglines] == ["one"]
        assert offset == len(line)


class TestReadNewLoglinesMeta:
    """The tail reader surfaces meta entries (titles) the static parser drops."""

    def test_ai_title_line_yields_meta_entry(self, tmp_path):
        p = tmp_path / "s.jsonl"
        data = _ai_title_line("Live name")
        p.write_bytes(data)
        loglines, offset = read_new_loglines(p, 0)
        assert loglines == [{"type": "ai-title", "title": "Live name"}]
        assert offset == len(data)


class TestFormatSseEvent:
    """SSE wire framing."""

    def test_basic(self):
        assert (
            format_sse_event("stats", {"prompts": 2})
            == 'event: stats\ndata: {"prompts": 2}\n\n'
        )

    def test_multiline_html_collapses_to_single_data_line(self):
        html = "<div>\n<p>hi</p>\n</div>"
        out = format_sse_event("append", {"html": html})

        assert out.startswith("event: append\n")
        assert out.endswith("\n\n")
        data_lines = [ln for ln in out.split("\n") if ln.startswith("data: ")]
        assert len(data_lines) == 1  # embedded newlines escaped by json.dumps
        payload = json.loads(data_lines[0][len("data: ") :])
        assert payload["html"] == html


class TestRenderLogline:
    """Single-entry HTML fragment for the live stream."""

    def test_user_entry(self):
        entry = {
            "type": "user",
            "timestamp": "T1",
            "message": {"role": "user", "content": "hello world"},
        }
        html = render_logline(entry)
        assert 'class="message user"' in html
        assert "hello world" in html

    def test_assistant_entry(self):
        entry = {
            "type": "assistant",
            "timestamp": "T2",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi back"}],
            },
        }
        html = render_logline(entry)
        assert 'class="message assistant"' in html
        assert "hi back" in html

    def test_empty_content_returns_empty(self):
        entry = {
            "type": "assistant",
            "timestamp": "T3",
            "message": {"role": "assistant", "content": []},
        }
        assert render_logline(entry) == ""

    def test_compact_summary_wrapped_in_details(self):
        entry = {
            "type": "user",
            "timestamp": "T4",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "resumed session"},
        }
        html = render_logline(entry)
        assert '<details class="continuation">' in html
        assert "Session continuation summary" in html
        assert "resumed session" in html


class TestIndexPrompt:
    """Detect real user prompts for the live TOC."""

    def test_user_with_text(self):
        entry = {
            "type": "user",
            "timestamp": "T",
            "message": {"role": "user", "content": "fix the bug"},
        }
        ok, preview = index_prompt(entry)
        assert ok is True
        assert preview == "fix the bug"

    def test_continuation_excluded(self):
        entry = {
            "type": "user",
            "timestamp": "T",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "resumed"},
        }
        assert index_prompt(entry) == (False, "")

    def test_stop_hook_feedback_excluded(self):
        entry = {
            "type": "user",
            "timestamp": "T",
            "message": {"role": "user", "content": "Stop hook feedback: blah"},
        }
        assert index_prompt(entry) == (False, "")

    def test_tool_result_message_excluded(self):
        entry = {
            "type": "user",
            "timestamp": "T",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "x", "content": "ok"}
                ],
            },
        }
        assert index_prompt(entry) == (False, "")

    def test_assistant_excluded(self):
        entry = {
            "type": "assistant",
            "timestamp": "T",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
            },
        }
        assert index_prompt(entry) == (False, "")

    def test_long_preview_truncated(self):
        entry = {
            "type": "user",
            "timestamp": "T",
            "message": {"role": "user", "content": "x" * 200},
        }
        ok, preview = index_prompt(entry)
        assert ok is True
        assert preview.endswith("...")
        assert len(preview) == 100


class TestLiveStats:
    """Cumulative counters for the live stats bar."""

    def test_accumulation_totals(self):
        entries = [
            {
                "type": "user",
                "timestamp": "T1",
                "message": {"role": "user", "content": "do it"},
            },
            {
                "type": "assistant",
                "timestamp": "T2",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "a",
                            "name": "Bash",
                            "input": {"command": "x"},
                        },
                        {"type": "tool_use", "id": "b", "name": "Edit", "input": {}},
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "T3",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "a",
                            "content": "[main abc1234] my commit\n",
                        }
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "T4",
                "message": {"role": "user", "content": "again"},
            },
        ]

        state = new_live_stats()
        for entry in entries:
            accumulate_live_stats(state, entry)

        assert live_stats_payload(state) == {
            "prompts": 2,
            "messages": 4,
            "tool_calls": 2,
            "commits": 1,
        }


class TestCardDataHelpers:
    """Pure helpers feeding the session info card (static embed + live SSE).

    Usage shape verified against real ~/.claude/projects files (Claude Code
    v2.1.x, 2026-06-10): message.usage = {"input_tokens": ..,
    "cache_creation_input_tokens": .., "cache_read_input_tokens": ..,
    "output_tokens": .., ...}. Context size = the three input-side numbers of
    the LATEST assistant entry; output accumulates.
    """

    def _assistant(self, text="ok", usage=None, ts="T"):
        msg = {"role": "assistant", "content": [{"type": "text", "text": text}]}
        if usage is not None:
            msg["usage"] = usage
        return {"type": "assistant", "timestamp": ts, "message": msg}

    def _user(self, text="hi", ts="T"):
        return {
            "type": "user",
            "timestamp": ts,
            "message": {"role": "user", "content": text},
        }

    def test_usage_latest_context_summed_output(self):
        loglines = [
            self._user(),
            self._assistant(
                usage={
                    "input_tokens": 2052,
                    "cache_creation_input_tokens": 41004,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 1360,
                }
            ),
            self._assistant(
                usage={
                    "input_tokens": 31,
                    "cache_creation_input_tokens": 1145,
                    "cache_read_input_tokens": 312625,
                    "output_tokens": 116,
                }
            ),
        ]
        assert compute_usage_totals(loglines) == {
            "context_tokens": 31 + 1145 + 312625,
            "output_tokens": 1360 + 116,
        }

    def test_usage_none_when_no_usage(self):
        """Web JSON exports carry no usage — the card hides the section."""
        loglines = [self._user(), self._assistant()]
        assert compute_usage_totals(loglines) == {
            "context_tokens": None,
            "output_tokens": 0,
        }

    def test_usage_trailing_entry_without_usage_keeps_context(self):
        loglines = [
            self._assistant(
                usage={
                    "input_tokens": 1,
                    "cache_creation_input_tokens": 2,
                    "cache_read_input_tokens": 3,
                    "output_tokens": 4,
                }
            ),
            self._assistant(),
        ]
        totals = compute_usage_totals(loglines)
        assert totals["context_tokens"] == 6
        assert totals["output_tokens"] == 4

    def test_usage_missing_fields_default_zero(self):
        loglines = [self._assistant(usage={"output_tokens": 5})]
        assert compute_usage_totals(loglines) == {
            "context_tokens": 0,
            "output_tokens": 5,
        }

    def test_snippet_last_text_block(self):
        loglines = [
            self._user(),
            self._assistant(text="first reply"),
            self._user(),
            self._assistant(text="final reply"),
        ]
        assert last_assistant_snippet(loglines) == "final reply"

    def test_snippet_skips_tool_only_assistant(self):
        tool_only = {
            "type": "assistant",
            "timestamp": "T",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Bash", "input": {}, "id": "t1"}
                ],
            },
        }
        loglines = [self._assistant(text="real text"), tool_only]
        assert last_assistant_snippet(loglines) == "real text"

    def test_snippet_collapses_whitespace_and_truncates(self):
        text = "line one\n\nline two   spaced " + "x" * 400
        loglines = [self._assistant(text=text)]
        s = last_assistant_snippet(loglines, max_length=50)
        assert len(s) <= 50
        assert "\n" not in s
        assert s.startswith("line one line two spaced")
        assert s.endswith("...")

    def test_snippet_none_when_no_assistant_text(self):
        assert last_assistant_snippet([self._user()]) is None

    def test_prompt_preview_collapses_and_caps(self):
        text = "a  b\nc " + "y" * 200
        p = prompt_preview(text)
        assert p.startswith("a b c")
        assert len(p) == 100
        assert p.endswith("...")

    def test_prompt_preview_short_text_unchanged(self):
        assert prompt_preview("fix the bug") == "fix the bug"


class TestResolveActiveSession:
    """Pick which session the live view tails."""

    def test_explicit_session_passthrough(self, tmp_path):
        target = tmp_path / "x" / "chosen.jsonl"
        assert resolve_active_session(tmp_path, session=str(target)) == Path(
            str(target)
        )

    def test_picks_newest_by_mtime(self, tmp_path):
        _write_session(tmp_path / "a" / "old.jsonl", _user_line("old"), 1000)
        _write_session(tmp_path / "b" / "new.jsonl", _user_line("new"), 2000)
        assert resolve_active_session(tmp_path) == tmp_path / "b" / "new.jsonl"

    def test_skips_agent_files(self, tmp_path):
        _write_session(tmp_path / "p" / "agent-sub.jsonl", _user_line("agent"), 3000)
        _write_session(tmp_path / "p" / "real.jsonl", _user_line("real"), 2000)
        assert resolve_active_session(tmp_path) == tmp_path / "p" / "real.jsonl"

    def test_skips_warmup(self, tmp_path):
        _write_session(tmp_path / "p" / "warm.jsonl", _user_line("warmup"), 3000)
        _write_session(tmp_path / "p" / "real.jsonl", _user_line("real"), 2000)
        assert resolve_active_session(tmp_path) == tmp_path / "p" / "real.jsonl"

    def test_keeps_fresh_no_summary_over_older(self, tmp_path):
        # A just-started session may have no summary yet; it must still win if newest.
        assistant_only = (
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "T",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "hi"}],
                    },
                }
            )
            + "\n"
        ).encode("utf-8")
        _write_session(tmp_path / "p" / "fresh.jsonl", assistant_only, 3000)
        _write_session(tmp_path / "p" / "real.jsonl", _user_line("real"), 2000)
        assert resolve_active_session(tmp_path) == tmp_path / "p" / "fresh.jsonl"

    def test_empty_dir_returns_none(self, tmp_path):
        assert resolve_active_session(tmp_path) is None

    def test_missing_dir_returns_none(self, tmp_path):
        assert resolve_active_session(tmp_path / "does-not-exist") is None


class TestLiveServer:
    """End-to-end: HTTP shell + SSE tail loop against a real localhost server."""

    def test_root_shell_and_live_stream(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_bytes(_user_line("first prompt", "2025-01-01T10:00:00.000Z"))

        server = create_live_server(p, poll_interval=0.02)
        port = server.server_address[1]
        t = _serve_in_thread(server)
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/", timeout=5)
            assert r.status_code == 200
            body = r.text
            assert 'id="messages"' in body
            assert 'id="stats-bar"' in body
            assert 'id="toc-list"' in body
            assert "EventSource" in body  # LIVE_JS wired in

            with httpx.stream(
                "GET", f"http://127.0.0.1:{port}/events", timeout=10
            ) as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"] == "text/event-stream"
                lines = resp.iter_lines()

                initial = _collect_sse(
                    lines, lambda evs: any(ev == "stats" for ev, _ in evs)
                )
                types = [ev for ev, _ in initial]
                assert "reset" in types
                assert "append" in types
                assert "prompt" in types
                assert "stats" in types
                assert _data_for(initial, "stats")[-1]["prompts"] == 1
                assert _data_for(initial, "prompt")[0]["preview"] == "first prompt"

                # Append a new prompt ONLY after draining the initial replay.
                with open(p, "ab") as f:
                    f.write(_user_line("second prompt", "2025-01-01T10:05:00.000Z"))
                    f.flush()
                    os.fsync(f.fileno())

                more = _collect_sse(
                    lines,
                    lambda evs: any(
                        ev == "stats" and json.loads(d)["prompts"] == 2 for ev, d in evs
                    ),
                )
                assert "append" in [ev for ev, _ in more]
                assert _data_for(more, "prompt")[-1]["preview"] == "second prompt"
                assert _data_for(more, "stats")[-1]["prompts"] == 2
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_truncation_triggers_reset(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_bytes(
            _user_line("first prompt one", "2025-01-01T10:00:00.000Z")
            + _user_line("second prompt two", "2025-01-01T10:01:00.000Z")
        )

        server = create_live_server(p, poll_interval=0.02)
        port = server.server_address[1]
        t = _serve_in_thread(server)
        try:
            with httpx.stream(
                "GET", f"http://127.0.0.1:{port}/events", timeout=10
            ) as resp:
                lines = resp.iter_lines()
                _collect_sse(
                    lines,
                    lambda evs: any(
                        ev == "stats" and json.loads(d)["prompts"] == 2 for ev, d in evs
                    ),
                )

                # Rewrite to a SMALLER file (size < offset) → compaction/truncation.
                p.write_bytes(_user_line("brand new", "2025-01-01T11:00:00.000Z"))

                def saw_reset_then_recount(evs):
                    saw_reset = False
                    for ev, d in evs:
                        if ev == "reset":
                            saw_reset = True
                        if (
                            saw_reset
                            and ev == "stats"
                            and json.loads(d)["prompts"] == 1
                        ):
                            return True
                    return False

                after = _collect_sse(lines, saw_reset_then_recount)
                assert any(ev == "reset" for ev, _ in after)
                assert _data_for(after, "stats")[-1]["prompts"] == 1
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_title_event_streams_and_dedupes(self, tmp_path):
        """Tab titles follow the session's ai-title: pre-titled shell, a title
        event on connect-replay and on change, no event for duplicates."""
        p = tmp_path / "s.jsonl"
        p.write_bytes(
            _user_line("first prompt", "2025-01-01T10:00:00.000Z")
            + _ai_title_line("First name")
        )

        server = create_live_server(p, poll_interval=0.02)
        port = server.server_address[1]
        t = _serve_in_thread(server)
        try:
            body = httpx.get(f"http://127.0.0.1:{port}/", timeout=5).text
            assert "First name" in body  # shell pre-titled server-side
            assert 'id="session-title"' in body

            with httpx.stream(
                "GET", f"http://127.0.0.1:{port}/events", timeout=10
            ) as resp:
                lines = resp.iter_lines()
                initial = _collect_sse(
                    lines, lambda evs: any(ev == "title" for ev, _ in evs)
                )
                assert _data_for(initial, "title") == [{"title": "First name"}]

                with open(p, "ab") as f:
                    f.write(_ai_title_line("Renamed"))
                    f.flush()
                    os.fsync(f.fileno())
                more = _collect_sse(
                    lines, lambda evs: any(ev == "title" for ev, _ in evs)
                )
                assert _data_for(more, "title") == [{"title": "Renamed"}]

                # A duplicate title must NOT re-emit; the trailing user line
                # provides a stats fence proving the tail consumed both lines.
                with open(p, "ab") as f:
                    f.write(_ai_title_line("Renamed"))
                    f.write(_user_line("second prompt", "2025-01-01T10:05:00.000Z"))
                    f.flush()
                    os.fsync(f.fileno())
                fenced = _collect_sse(
                    lines,
                    lambda evs: any(
                        ev == "stats" and json.loads(d)["prompts"] == 2 for ev, d in evs
                    ),
                )
                assert _data_for(fenced, "title") == []
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_disconnect_then_shutdown_is_clean(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_bytes(_user_line("hi", "2025-01-01T10:00:00.000Z"))

        server = create_live_server(p, poll_interval=0.02)
        port = server.server_address[1]
        t = _serve_in_thread(server)

        # Connect, read one line to confirm the stream is live, then disconnect.
        with httpx.stream("GET", f"http://127.0.0.1:{port}/events", timeout=5) as resp:
            for _ in resp.iter_lines():
                break

        # Teardown must complete promptly despite the (now dead) handler thread.
        server.shutdown()
        server.server_close()
        t.join(timeout=5)
        assert not t.is_alive()


class TestWatchCommand:
    """CLI wiring for `watch`. The blocking serve loop is stubbed so the runner
    doesn't hang; the tail loop itself is covered by TestLiveServer."""

    def test_help_lists_options(self):
        result = CliRunner().invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--session" in result.output
        assert "--pick" in result.output
        assert "--no-open" in result.output

    def test_no_session_found_returns_without_serving(self, tmp_path):
        # Empty source dir → nothing to tail → graceful early return (no hang).
        result = CliRunner().invoke(cli, ["watch", "--source", str(tmp_path)])
        assert result.exit_code == 0
        assert "No active session" in result.output

    def test_serves_resolved_newest_session(
        self, tmp_path, monkeypatch, mock_webbrowser_open
    ):
        _write_session(tmp_path / "p" / "live.jsonl", _user_line("hello"), 2000)
        monkeypatch.setattr(
            "claude_code_transcripts._LiveServer.serve_forever", lambda self: None
        )

        result = CliRunner().invoke(cli, ["watch", "--source", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "live.jsonl" in result.output
        assert any("http://127.0.0.1:" in url for url in mock_webbrowser_open)

    def test_no_open_skips_browser(self, tmp_path, monkeypatch, mock_webbrowser_open):
        _write_session(tmp_path / "p" / "live.jsonl", _user_line("hello"), 2000)
        monkeypatch.setattr(
            "claude_code_transcripts._LiveServer.serve_forever", lambda self: None
        )

        result = CliRunner().invoke(
            cli, ["watch", "--source", str(tmp_path), "--no-open"]
        )

        assert result.exit_code == 0, result.output
        assert mock_webbrowser_open == []

    def test_pick_uses_questionary_selection(
        self, tmp_path, monkeypatch, mock_webbrowser_open
    ):
        chosen = tmp_path / "p" / "chosen.jsonl"
        _write_session(chosen, _user_line("pick me"), 2000)
        _write_session(tmp_path / "p" / "newer.jsonl", _user_line("newer"), 9000)

        class _FakePrompt:
            def ask(self):
                return chosen

        monkeypatch.setattr(
            "claude_code_transcripts.questionary.select",
            lambda *a, **k: _FakePrompt(),
        )
        monkeypatch.setattr(
            "claude_code_transcripts._LiveServer.serve_forever", lambda self: None
        )

        result = CliRunner().invoke(cli, ["watch", "--pick", "--source", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "chosen.jsonl" in result.output  # picked, not the newer one

    def test_help_lists_limit(self):
        result = CliRunner().invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--limit" in result.output

    def test_pick_rows_match_local_format(
        self, tmp_path, monkeypatch, mock_webbrowser_open
    ):
        """--pick rows carry the same branch/project/command columns as `local`.

        The old watch picker printed only date · size · summary, which made
        eight identical "/plan ..." sessions indistinguishable.
        """
        line = {
            "type": "user",
            "timestamp": "T",
            "gitBranch": "dti/x",
            "message": {
                "role": "user",
                "content": (
                    "<command-message>plan</command-message>\n"
                    "<command-name>/plan</command-name>\n"
                    "<command-args>build the widget</command-args>"
                ),
            },
        }
        _write_session(
            tmp_path / "D--projects-devjig" / "s.jsonl",
            (json.dumps(line) + "\n").encode("utf-8"),
            2000,
        )

        captured = {}

        def fake_select(message, choices=None, **kwargs):
            captured["choices"] = choices

            class _P:
                def ask(self):
                    return choices[0].value

            return _P()

        monkeypatch.setattr("claude_code_transcripts.questionary.select", fake_select)
        monkeypatch.setattr(
            "claude_code_transcripts._LiveServer.serve_forever", lambda self: None
        )

        result = CliRunner().invoke(cli, ["watch", "--pick", "--source", str(tmp_path)])

        assert result.exit_code == 0, result.output
        row = captured["choices"][0].title
        assert "[dti/x]" in row
        assert "devjig" in row  # decoded project display name
        assert "/plan" in row
        assert "build the widget" in row

    def test_pick_respects_limit(self, tmp_path, monkeypatch, mock_webbrowser_open):
        for i in range(3):
            _write_session(
                tmp_path / "p" / f"s{i}.jsonl", _user_line(f"prompt {i}"), 2000 + i
            )

        captured = {}

        def fake_select(message, choices=None, **kwargs):
            captured["choices"] = choices

            class _P:
                def ask(self):
                    return choices[0].value

            return _P()

        monkeypatch.setattr("claude_code_transcripts.questionary.select", fake_select)
        monkeypatch.setattr(
            "claude_code_transcripts._LiveServer.serve_forever", lambda self: None
        )

        result = CliRunner().invoke(
            cli, ["watch", "--pick", "--limit", "2", "--source", str(tmp_path)]
        )

        assert result.exit_code == 0, result.output
        assert len(captured["choices"]) == 2
