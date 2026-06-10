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
from pathlib import Path

from claude_code_transcripts import (
    scan_session_metadata,
    SessionMetadata,
    format_session_choice,
    build_session_choices,
    strip_recap_suffix,
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

    def test_ai_title_last_wins(self, tmp_path):
        """Claude Code appends ai-title lines as the auto-title evolves; the
        LAST one is the session's current name (what `claude --resume` shows).

        Shape verified against real ~/.claude/projects files (Claude Code
        v2.1.x, 2026-06-10):
        {"type":"ai-title","aiTitle":"...","sessionId":"..."}
        """
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                _user("plain prose prompt"),
                {"type": "ai-title", "aiTitle": "Early title", "sessionId": "abc"},
                {
                    "type": "ai-title",
                    "aiTitle": "Fix database refresh script column errors",
                    "sessionId": "abc",
                },
            ],
        )
        meta = scan_session_metadata(f)
        assert meta.ai_title == "Fix database refresh script column errors"

    def test_ai_title_absent_is_none(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(f, [_user("plain prose prompt")])
        assert scan_session_metadata(f).ai_title is None

    def test_title_prefers_ai_title_over_summary(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                _user("plain prose prompt"),
                {"type": "ai-title", "aiTitle": "The real name", "sessionId": "abc"},
            ],
        )
        assert scan_session_metadata(f).title == "The real name"

    def test_title_falls_back_to_summary(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(f, [_user("plain prose prompt")])
        assert scan_session_metadata(f).title == "plain prose prompt"

    def test_ai_title_does_not_displace_summary(self, tmp_path):
        """ai_title and summary are independent: the picker keeps showing the
        prompt-derived summary while title consumers get the AI name."""
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                {"type": "ai-title", "aiTitle": "Name", "sessionId": "abc"},
                _user("plain prose prompt"),
            ],
        )
        meta = scan_session_metadata(f)
        assert meta.summary == "plain prose prompt"
        assert meta.ai_title == "Name"


class TestTitleChanges:
    """ai-title lines carry no timestamp, so changes are anchored to the
    latest user-line timestamp seen at the moment the change was read; the
    chapter divider goes before the first prompt AFTER that anchor."""

    def _title(self, t):
        return {"type": "ai-title", "aiTitle": t, "sessionId": "s"}

    def _user_at(self, text, ts):
        obj = _user(text)
        obj["timestamp"] = ts
        return obj

    def test_stable_title_no_changes(self, tmp_path):
        """Real sessions repeat the SAME title dozens of times (43 observed
        in one file) — repeats are not changes."""
        f = tmp_path / "s.jsonl"
        entries = [_user("start the work")]
        entries += [self._title("One stable name")] * 43
        _write_jsonl(f, entries)
        meta = scan_session_metadata(f)
        assert meta.title_changes == ()
        assert meta.ai_title == "One stable name"

    def test_change_anchored_to_latest_user_timestamp(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                self._user_at("first prompt", "T1"),
                self._title("Name A"),
                self._user_at("second prompt", "T2"),
                self._title("Name B"),
            ],
        )
        assert scan_session_metadata(f).title_changes == (("T2", "Name B"),)

    def test_consecutive_duplicates_deduped_distinct_rerecorded(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                self._user_at("p1", "T1"),
                self._title("A"),
                self._title("A"),
                self._title("B"),
                self._title("B"),
                self._title("A"),
            ],
        )
        assert scan_session_metadata(f).title_changes == (("T1", "B"), ("T1", "A"))

    def test_change_before_any_user_anchors_empty(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(f, [self._title("A"), self._title("B"), _user("hi")])
        assert scan_session_metadata(f).title_changes == (("", "B"),)

    def test_chapter_titles_truncated(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f, [self._user_at("p", "T1"), self._title("A"), self._title("X" * 300)]
        )
        meta = scan_session_metadata(f, max_length=40)
        ((_, title),) = meta.title_changes
        assert len(title) <= 40


def _away_summary(content, ts="2026-06-10T13:53:32.990Z"):
    """One away_summary system entry — Claude Code's "※ recap:" text.

    Shape verified against real ~/.claude/projects files (Claude Code v2.1.x,
    2026-06-10): {"type":"system","subtype":"away_summary","content":"...",
    "isMeta":false,"timestamp":"..."}
    """
    return {
        "type": "system",
        "subtype": "away_summary",
        "content": content,
        "isMeta": False,
        "timestamp": ts,
    }


class TestStripRecapSuffix:
    """Some away_summary contents carry a UI hint trailer to strip for display."""

    def test_strips_trailing_hint(self):
        assert (
            strip_recap_suffix("Work is done. (disable recaps in /config)")
            == "Work is done."
        )

    def test_no_suffix_untouched(self):
        assert strip_recap_suffix("Work is done.") == "Work is done."

    def test_mid_text_mention_untouched(self):
        text = "You can (disable recaps in /config) at any time, then continue."
        assert strip_recap_suffix(text) == text


class TestScanSessionMetadataRecap:
    """The latest away_summary is the session's current recap."""

    def test_recap_last_wins(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                _user("plain prose prompt"),
                _away_summary("Old recap of earlier work."),
                _away_summary("Slice 7 of the plan is shipped; nothing pending."),
            ],
        )
        meta = scan_session_metadata(f)
        assert meta.recap == "Slice 7 of the plan is shipped; nothing pending."

    def test_recap_suffix_stripped(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                _user("plain prose prompt"),
                _away_summary("All tests green. (disable recaps in /config)"),
            ],
        )
        assert scan_session_metadata(f).recap == "All tests green."

    def test_recap_absent_is_none(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(f, [_user("plain prose prompt")])
        assert scan_session_metadata(f).recap is None

    def test_other_system_subtypes_ignored(self, tmp_path):
        f = tmp_path / "s.jsonl"
        _write_jsonl(
            f,
            [
                _user("plain prose prompt"),
                {"type": "system", "subtype": "turn_duration", "timestamp": "T"},
                {"type": "system", "subtype": "local_command", "content": "ran /foo"},
                {"type": "system", "subtype": "compact_boundary", "timestamp": "T"},
            ],
        )
        assert scan_session_metadata(f).recap is None

    def test_recap_not_truncated_by_max_length(self, tmp_path):
        """Recaps feed the info card, not picker rows — keep them whole."""
        long_recap = "A detailed recap sentence. " * 20
        f = tmp_path / "s.jsonl"
        _write_jsonl(f, [_user("hi"), _away_summary(long_recap)])
        meta = scan_session_metadata(f, max_length=40)
        assert meta.recap == long_recap.strip()


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


def _make_session(projects_dir, project_folder, name, entries):
    """Write one session JSONL under a Claude-style encoded project folder."""
    proj = projects_dir / project_folder
    proj.mkdir(parents=True, exist_ok=True)
    f = proj / name
    _write_jsonl(f, entries)
    return f


class TestBuildSessionChoices:
    """Tests for build_session_choices — the shared local/watch picker rows.

    Both the `local` and `watch --pick` pickers must render identical rows
    (date · size · [branch] · project · command · summary), built from ONE
    metadata scan per file.
    """

    def test_choice_rows_use_shared_format(self, tmp_path):
        f = _make_session(
            tmp_path,
            "D--projects-devjig",
            "abc.jsonl",
            [_user(_command("/plan", "build the widget"), branch="dti/x")],
        )
        choices = build_session_choices(tmp_path)
        assert len(choices) == 1
        assert choices[0].value == f
        row = choices[0].title
        assert "[dti/x]" in row
        assert "devjig" in row  # decoded project display name
        assert "/plan" in row
        assert "build the widget" in row

    def test_scans_each_file_once(self, tmp_path, monkeypatch):
        """The old local picker scanned every file twice (find + per-row)."""
        import claude_code_transcripts as cct

        for i in range(2):
            _make_session(tmp_path, "proj", f"s{i}.jsonl", [_user(f"prompt {i}")])

        calls = []
        real = cct.scan_session_metadata

        def counting(path, *args, **kwargs):
            calls.append(Path(path))
            return real(path, *args, **kwargs)

        monkeypatch.setattr(cct, "scan_session_metadata", counting)
        cct.build_session_choices(tmp_path)
        assert len(calls) == 2

    def test_respects_limit(self, tmp_path):
        for i in range(5):
            _make_session(tmp_path, "proj", f"s{i}.jsonl", [_user(f"prompt {i}")])
        assert len(build_session_choices(tmp_path, limit=3)) == 3

    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert build_session_choices(tmp_path) == []
        assert build_session_choices(tmp_path / "missing") == []
