from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scaling_evolve.providers.agent.session_log import build_session_log_markdown


def _rollout(transcript_path: Path, *, session_id: str, effort: str | None = None):
    execution: dict[str, object] = {"driver": "claude_code"}
    if effort is not None:
        execution["effort_level"] = effort
    return SimpleNamespace(
        state=SimpleNamespace(
            session_id=session_id,
            metadata={
                "driver_execution": execution,
                "provider_transcript_path": str(transcript_path),
            },
        ),
        usage=SimpleNamespace(input_tokens=20, output_tokens=8, cache_read_tokens=5),
        summary="done",
    )


def test_build_session_log_markdown_for_cumulative_stream_json_transcript(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "session-s1.jsonl"
    cwd = "/runs/solver_workspaces/20260927_step_3_abcd"
    records = [
        {
            "type": "eve.rollout",
            "timestamp": "2026-09-27T10:00:00Z",
            "session_id": "s1",
            "instruction": "Read README.md first.",
            "model": "claude-opus-5-5",
            "effort": "xhigh",
            "role": "solver",
            "cwd": cwd,
        },
        {
            "type": "system",
            "subtype": "init",
            "session_id": "s1",
            "cwd": cwd,
            "model": "claude-opus-5-5",
            "apiKeySource": "none",
        },
        {
            "type": "assistant",
            "session_id": "s1",
            "parent_tool_use_id": None,
            "message": {
                "id": "msg_1",
                "model": "claude-opus-5-5",
                "content": [
                    {"type": "text", "text": "Delegating the check."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Task", "input": {}},
                ],
            },
        },
        {
            "type": "assistant",
            "session_id": "s1",
            "parent_tool_use_id": "toolu_1",
            "message": {
                "id": "msg_sub",
                "content": [{"type": "text", "text": "SUBAGENT CHATTER"}],
            },
        },
        {
            "type": "user",
            "session_id": "s1",
            "parent_tool_use_id": None,
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "PASS"},
                ],
            },
        },
        {"type": "rate_limit_event", "session_id": "s1", "rate_limit_info": {"status": "allowed"}},
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "session_id": "s1",
            "result": "First rollout done.",
        },
        {
            "type": "eve.usage_limit_wait",
            "session_id": "s1",
            "waited_seconds": 60,
            "instruction": "continue",
        },
        {
            "type": "eve.rollout",
            "timestamp": "2026-09-27T11:00:00Z",
            "session_id": "s1",
            "instruction": "Fix the boundary violation.",
            "model": "claude-opus-5-5",
            "effort": "xhigh",
            "role": "solver",
            "cwd": cwd,
        },
        {
            "type": "assistant",
            "session_id": "s2-after-resume",
            "message": {"id": "msg_2", "content": [{"type": "text", "text": "Repaired."}]},
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "session_id": "s2-after-resume",
            "result": "Repaired the boundary.",
        },
    ]
    transcript_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    markdown = build_session_log_markdown([_rollout(transcript_path, session_id="s1")])

    assert markdown is not None
    assert "# Session Log - solver (iter 3)" in markdown
    assert "- **Provider**: claude_code" in markdown
    assert "- **Model**: claude-opus-5-5 (effort=xhigh)" in markdown
    assert "Read README.md first." in markdown
    assert "Fix the boundary violation." in markdown
    assert "Delegating the check." in markdown
    assert "**tool**: `Task`" in markdown
    assert "Repaired." in markdown
    assert "Repaired the boundary." in markdown
    assert "SUBAGENT CHATTER" not in markdown
