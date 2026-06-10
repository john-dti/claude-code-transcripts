"""Tests for artifact extraction: deep-linkable insights, substantial
thinking, plans, and per-prompt completion replies feeding the session card
and index navigation.

Shapes mirror real ~/.claude/projects JSONL (Claude Code v2.1.x, 2026-06-10):
★ Insight marker blocks in assistant text; ExitPlanMode tool_use with modern
{"allowedPrompts": [...]} input (plan text lives in the tool result; legacy
sessions carried input.plan markdown).
"""

import json

from claude_code_transcripts import (
    ARTIFACT_ICONS,
    LONG_TEXT_THRESHOLD,
    artifact_label,
    is_insight_text,
    insight_label,
    plan_label,
    iter_block_artifacts,
    extract_conversation_artifacts,
    extract_entry_artifacts,
    render_message,
)

INSIGHT_TEXT = (
    "`★ Insight ─────────────────────────────────────`\n"
    "- The scan is single-pass for a reason\n"
    "- Anchors must come from one enumeration\n"
    "`─────────────────────────────────────────────────`"
)

MSG_TS = "2025-01-01T10:00:30.000Z"
MSG_ID = "msg-2025-01-01T10-00-30-000Z"


def _assistant_msg(blocks):
    return {"role": "assistant", "content": blocks}


def _assistant_tuple(blocks, ts=MSG_TS):
    return ("assistant", json.dumps(_assistant_msg(blocks)), ts)


def _text(t):
    return {"type": "text", "text": t}


def _thinking(t):
    return {"type": "thinking", "thinking": t}


def _plan_use(tool_input=None):
    return {
        "type": "tool_use",
        "name": "ExitPlanMode",
        "input": tool_input if tool_input is not None else {"allowedPrompts": []},
        "id": "t-plan",
    }


class TestLabels:
    def test_artifact_label_collapses_and_caps(self):
        label = artifact_label("line  one\nline two " + "x" * 100)
        assert "\n" not in label
        assert "  " not in label
        assert len(label) <= 60
        assert label.endswith("...")

    def test_insight_label_first_line_after_marker(self):
        assert insight_label(INSIGHT_TEXT) == "The scan is single-pass for a reason"

    def test_insight_label_falls_back_to_text(self):
        text = "`★ Insight ──`\n`────`"  # nothing usable after the marker
        assert insight_label(text) == artifact_label(text)

    def test_plan_label_modern_format_is_plan_presented(self):
        assert plan_label({"allowedPrompts": [{"tool": "Bash"}]}) == "Plan presented"

    def test_plan_label_legacy_uses_first_heading(self):
        label = plan_label({"plan": "intro\n## Big Migration Plan\nbody"})
        assert "Big Migration Plan" in label

    def test_is_insight_text(self):
        assert is_insight_text(INSIGHT_TEXT)
        assert not is_insight_text("plain reply")

    def test_icon_map_covers_exactly_four_types(self):
        assert set(ARTIFACT_ICONS) == {"insight", "thinking", "plan", "completion"}


class TestIterBlockArtifacts:
    def test_kinds_and_thresholds(self):
        blocks = [
            _thinking("x" * (LONG_TEXT_THRESHOLD - 1)),  # b0: under threshold
            _thinking("y" * LONG_TEXT_THRESHOLD),  # b1: at threshold
            _text(INSIGHT_TEXT),  # b2: insight
            _text("plain reply"),  # b3: completion candidate
            _plan_use(),  # b4: plan
            {"type": "tool_use", "name": "Grep", "input": {}, "id": "t2"},  # b5: no
        ]
        got = list(iter_block_artifacts(_assistant_msg(blocks), "msg-X"))
        kinds_anchors = [(k, a) for k, a, _ in got]
        assert kinds_anchors == [
            ("thinking", "msg-X-b1"),
            ("insight", "msg-X-b2"),
            ("text", "msg-X-b3"),
            ("plan", "msg-X-b4"),
        ]

    def test_empty_message_yields_nothing(self):
        assert list(iter_block_artifacts(_assistant_msg([]), "msg-X")) == []


class TestExtractConversationArtifacts:
    def test_completion_is_last_text_block(self):
        groups = [
            (
                1,
                [
                    ("user", json.dumps({"role": "user", "content": "go"}), "T0"),
                    _assistant_tuple([_text("working on it"), _text("all done")]),
                ],
            )
        ]
        arts = extract_conversation_artifacts(groups)
        assert arts == [
            {
                "type": "completion",
                "label": "all done",
                "anchor": f"{MSG_ID}-b1",
                "page": 1,
            }
        ]

    def test_insight_as_last_text_retyped_in_place_keeps_label(self):
        groups = [
            (
                1,
                [
                    _assistant_tuple(
                        [_thinking("z" * LONG_TEXT_THRESHOLD), _text(INSIGHT_TEXT)]
                    )
                ],
            )
        ]
        arts = extract_conversation_artifacts(groups)
        assert len(arts) == 2  # thinking + the retyped insight; NOT three
        assert arts[0]["type"] == "thinking"
        assert arts[1]["type"] == "completion"
        assert arts[1]["label"] == "The scan is single-pass for a reason"
        assert arts[1]["anchor"] == f"{MSG_ID}-b1"

    def test_continuation_pages_attributed(self):
        groups = [
            (
                1,
                [
                    _assistant_tuple(
                        [_text(INSIGHT_TEXT)], ts="2025-01-01T10:00:30.000Z"
                    )
                ],
            ),
            (
                2,
                [
                    _assistant_tuple(
                        [_text("final words")], ts="2025-01-01T10:09:30.000Z"
                    )
                ],
            ),
        ]
        arts = extract_conversation_artifacts(groups)
        assert [a["page"] for a in arts] == [1, 2]
        assert arts[1]["type"] == "completion"
        assert arts[1]["anchor"].startswith("msg-2025-01-01T10-09-30-000Z")

    def test_no_assistant_text_no_completion(self):
        groups = [
            (
                1,
                [
                    _assistant_tuple(
                        [{"type": "tool_use", "name": "Grep", "input": {}, "id": "t"}]
                    )
                ],
            )
        ]
        assert extract_conversation_artifacts(groups) == []

    def test_empty_returns_empty(self):
        assert extract_conversation_artifacts([]) == []


class TestExtractEntryArtifacts:
    def test_immediate_artifacts_and_last_text(self):
        entry = {
            "type": "assistant",
            "timestamp": MSG_TS,
            "message": _assistant_msg(
                [
                    _thinking("z" * LONG_TEXT_THRESHOLD),
                    _text(INSIGHT_TEXT),
                    _text("wrapping up"),
                ]
            ),
        }
        immediate, last_text = extract_entry_artifacts(entry)
        assert [a["type"] for a in immediate] == ["thinking", "insight"]
        assert immediate[0]["id"] == f"{MSG_ID}-b0"
        assert last_text == {"id": f"{MSG_ID}-b2", "label": "wrapping up"}

    def test_insight_as_final_text_is_last_text_too(self):
        entry = {
            "type": "assistant",
            "timestamp": MSG_TS,
            "message": _assistant_msg([_text(INSIGHT_TEXT)]),
        }
        immediate, last_text = extract_entry_artifacts(entry)
        assert [a["type"] for a in immediate] == ["insight"]
        assert last_text["id"] == f"{MSG_ID}-b0"
        assert last_text["label"] == "The scan is single-pass for a reason"

    def test_tool_only_entry_none(self):
        entry = {
            "type": "assistant",
            "timestamp": MSG_TS,
            "message": _assistant_msg(
                [{"type": "tool_use", "name": "Grep", "input": {}, "id": "t"}]
            ),
        }
        assert extract_entry_artifacts(entry) == ([], None)

    def test_non_assistant_entry_none(self):
        entry = {
            "type": "user",
            "timestamp": MSG_TS,
            "message": {"role": "user", "content": "hello"},
        }
        assert extract_entry_artifacts(entry) == ([], None)


class TestAnchorDrift:
    def test_extracted_anchors_exist_in_rendered_html(self):
        """The drift guard: every anchor extraction produces must exist as an
        id in the rendered HTML for the same message."""
        blocks = [
            _thinking("z" * LONG_TEXT_THRESHOLD),
            _text(INSIGHT_TEXT),
            _plan_use(),
            _text("the completion"),
        ]
        message_json = json.dumps(_assistant_msg(blocks))
        rendered = render_message("assistant", message_json, MSG_TS)
        groups = [(1, [("assistant", message_json, MSG_TS)])]
        for art in extract_conversation_artifacts(groups):
            assert f'id="{art["anchor"]}"' in rendered
