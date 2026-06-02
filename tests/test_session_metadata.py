"""Tests for unified session-metadata extraction and picker-row formatting.

scan_session_metadata() does a single pass over a JSONL session and returns a
SessionMetadata(summary, branch, command, from_control_fallback). It powers the
session picker and the HTML archive listing. The behaviors under test:

- Non-demarcating control commands (/effort, /clear, ...) are skipped so the
  real task command (e.g. /plan) becomes the title — but a control-command-only
  session still gets a title (fallback) instead of being dropped.
- gitBranch is captured to disambiguate otherwise-identical sessions.
- The summary holds the command *args body* (the real prose); the command name
  is surfaced separately.
"""

import json

from claude_code_transcripts import (
    scan_session_metadata,
    SessionMetadata,
    format_session_choice,
)


def _user(content, branch=None, is_meta=False):
    """Build one user JSONL entry dict (mirrors Claude Code's on-disk shape)."""
    obj = {
        "type": "user",
        "timestamp": "2026-05-27T14:49:26Z",
        "message": {"role": "user", "content": content},
    }
    if branch is not None:
        obj["gitBranch"] = branch
    if is_meta:
        obj["isMeta"] = True
    return obj


def _command(name, args):
    """Render the <command-*> wrapper Claude Code records for a typed slash command."""
    return (
        f"<command-message>{name.lstrip('/')}</command-message>\n"
        f"<command-name>{name}</command-name>\n"
        f"<command-args>{args}</command-args>"
    )


def _write_jsonl(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


class TestScanSessionMetadata:
    """Tests for scan_session_metadata - the single-pass metadata extractor."""

    def test_returns_dataclass_with_all_fields(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(f, [_user(_command("/plan", "do the thing"), branch="main")])
        meta = scan_session_metadata(f)
        assert isinstance(meta, SessionMetadata)
        assert meta.summary
        assert meta.branch == "main"
        assert meta.command == "/plan"
        assert meta.from_control_fallback is False

    def test_skips_control_command_and_picks_task_command(self, tmp_path):
        """A leading /effort toggle must not become the title; /plan should win."""
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                _user(
                    "<local-command-caveat>noise</local-command-caveat>", is_meta=True
                ),
                _user(_command("/effort", "ultracode"), branch="dti/integration"),
                _user("<local-command-stdout>Set effort level</local-command-stdout>"),
                _user(_command("/plan", "build the widget carefully")),
            ],
        )
        meta = scan_session_metadata(f)
        assert "build the widget carefully" in meta.summary
        assert meta.command == "/plan"
        assert "effort" not in meta.summary
        assert "ultracode" not in meta.summary
        assert meta.from_control_fallback is False
        assert meta.branch == "dti/integration"

    def test_control_only_session_falls_back(self, tmp_path):
        """A session that is ONLY a control command still gets a title (not dropped)."""
        f = tmp_path / "s.jsonl"
        _write_jsonl(f, [_user(_command("/clear", ""))])
        meta = scan_session_metadata(f)
        assert meta.command == "/clear"
        assert meta.from_control_fallback is True
        assert meta.summary != "(no summary)"
        assert "/clear" in meta.summary

    def test_task_command_summary_is_body_only(self, tmp_path):
        """Summary is the args body, not a 'name: body' concatenation."""
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                _user(
                    _command("/morningly", "Loving Father, please direct my thinking.")
                )
            ],
        )
        meta = scan_session_metadata(f)
        assert meta.summary.startswith("Loving Father")
        assert not meta.summary.startswith("/morningly")
        assert meta.command == "/morningly"

    def test_captures_git_branch(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(f, [_user("plain prose prompt", branch="dti/watch")])
        assert scan_session_metadata(f).branch == "dti/watch"

    def test_branch_none_when_absent(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(f, [_user("plain prose prompt")])
        assert scan_session_metadata(f).branch is None

    def test_explicit_summary_line_wins(self, tmp_path):
        """A type==summary line beats the first user message; branch still captured."""
        f = tmp_path / "s.jsonl"
        f.write_text(
            json.dumps({"type": "summary", "summary": "Explicit session title"})
            + "\n"
            + json.dumps(_user("some later prose", branch="feature/x"))
            + "\n",
            encoding="utf-8",
        )
        meta = scan_session_metadata(f)
        assert meta.summary == "Explicit session title"
        assert meta.branch == "feature/x"

    def test_non_command_angle_bracket_skipped(self, tmp_path):
        """<system-reminder> and friends are skipped; real prose wins, command is None."""
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                _user("<system-reminder>boilerplate</system-reminder>"),
                _user("actual user prompt"),
            ],
        )
        meta = scan_session_metadata(f)
        assert meta.summary == "actual user prompt"
        assert meta.command is None

    def test_empty_file_is_no_summary(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        meta = scan_session_metadata(f)
        assert meta.summary == "(no summary)"
        assert meta.branch is None
        assert meta.command is None


# A fixed epoch so the rendered date is stable across machines/timezones; the
# assertions below only check the variable columns, not the date itself.
_MTIME = 1700000000.0


class TestFormatSessionChoice:
    """Tests for the shared picker-row formatter used by the session picker."""

    def test_includes_branch_project_command(self):
        meta = SessionMetadata(
            summary="do the precise thing",
            branch="dti/integration",
            command="/plan",
            from_control_fallback=False,
        )
        row = format_session_choice(meta, _MTIME, 43008, "claude-code-transcripts")
        assert "[dti/integration]" in row
        assert "claude-code-transcripts" in row
        assert "/plan" in row
        assert "do the precise thing" in row
        assert "42 KB" in row  # 43008 bytes / 1024

    def test_omits_columns_when_absent(self):
        meta = SessionMetadata(
            summary="plain prose prompt",
            branch=None,
            command=None,
            from_control_fallback=False,
        )
        row = format_session_choice(meta, _MTIME, 1024, "proj")
        assert "[" not in row  # no empty branch brackets
        assert "plain prose prompt" in row
        assert "  proj" in row

    def test_control_fallback_blanks_summary_keeps_command(self):
        """A /clear-only row shows the command once, not duplicated in the summary."""
        meta = SessionMetadata(
            summary="/clear",
            branch=None,
            command="/clear",
            from_control_fallback=True,
        )
        row = format_session_choice(meta, _MTIME, 1024, "proj")
        assert row.count("/clear") == 1

    def test_keeps_summary_when_command_has_body(self):
        """A control-fallback command that carries a body keeps the body, not blanked.

        Only an exact summary==command duplicate (a bare /clear) is suppressed.
        """
        meta = SessionMetadata(
            summary="condense the discussion so far",
            branch=None,
            command="/compact",
            from_control_fallback=True,
        )
        row = format_session_choice(meta, _MTIME, 1024, "proj")
        assert "/compact" in row
        assert "condense the discussion so far" in row

    def test_truncates_long_summary(self):
        meta = SessionMetadata(
            summary="x" * 200,
            branch=None,
            command=None,
            from_control_fallback=False,
        )
        row = format_session_choice(meta, _MTIME, 1024, "proj", summary_width=44)
        assert "..." in row
        assert "x" * 200 not in row
