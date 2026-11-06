from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scaling_evolve.providers.agent.drivers._metadata import TokenPricing
from scaling_evolve.providers.agent.drivers.base import SessionSeed, SessionWorkspaceLease
from scaling_evolve.providers.agent.drivers.claude_code import (
    ClaudeCodeSessionDriver,
)

_LIVE_CLAUDE_CODE = os.environ.get("SCALING_EVOLVE_RUN_LIVE_CLAUDE_CODE_TESTS") == "1"
# Live tests use the `/login` session unless a `claude setup-token` token is exported.
_LIVE_PROVIDER_ENV = {
    key: os.environ[key] for key in ("CLAUDE_CODE_OAUTH_TOKEN",) if os.environ.get(key)
}
_PRICING_TABLE = {
    "claude-sonnet-5-5": TokenPricing(
        input_per_million=2.0,
        output_per_million=10.0,
        cache_read_per_million=0.2,
    ),
}


def _init_git_repo(worktree: Path, *, initial_contents: str = "VALUE = 1\n") -> None:
    worktree.mkdir()
    subprocess.run(["git", "-C", str(worktree), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    (worktree / "candidate.py").write_text(initial_contents, encoding="utf-8")
    subprocess.run(["git", "-C", str(worktree), "add", "candidate.py"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "init"], check=True)


def _workspace_lease(worktree: Path, *, workspace_id: str = "attempt-1") -> SessionWorkspaceLease:
    return SessionWorkspaceLease(
        workspace_id=workspace_id,
        target_repo_root=str(worktree),
        workspace_root=str(worktree),
        session_cwd=str(worktree),
    )


def _stream(
    *,
    session_id: str,
    text: str,
    subtype: str = "success",
    is_error: bool = False,
    num_turns: int = 1,
    total_cost_usd: float | None = None,
) -> str:
    result: dict[str, object] = {
        "type": "result",
        "subtype": subtype,
        "is_error": is_error,
        "session_id": session_id,
        "num_turns": num_turns,
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 2,
            "cache_creation_input_tokens": 4,
            "output_tokens": 3,
        },
    }
    if subtype == "success":
        result["result"] = text
    if total_cost_usd is not None:
        result["total_cost_usd"] = total_cost_usd
    events = [
        {"type": "system", "subtype": "init", "session_id": session_id, "model": "m"},
        {
            "type": "assistant",
            "session_id": session_id,
            "message": {"id": "msg-1", "content": [{"type": "text", "text": text}]},
        },
        result,
    ]
    return "".join(json.dumps(event) + "\n" for event in events)


def test_claude_code_parse_stdout_extracts_session_id_usage_and_summary() -> None:
    stdout = "\n".join(
        [
            '{"type":"system","subtype":"init","session_id":"session-123"}',
            (
                '{"type":"assistant","session_id":"session-123","message":{"id":"msg-1",'
                '"content":[{"type":"text","text":"Inspecting the file."}]}}'
            ),
            (
                '{"type":"assistant","session_id":"session-123","parent_tool_use_id":"tool-1",'
                '"message":{"id":"sub-1","content":[{"type":"text","text":"subagent"}]}}'
            ),
            (
                '{"type":"assistant","session_id":"session-123","message":{"id":"msg-2",'
                '"content":[{"type":"text","text":"Mutation finished."}]}}'
            ),
            (
                '{"type":"result","subtype":"success","is_error":false,'
                '"session_id":"session-123","result":"Mutation finished.","num_turns":2,'
                '"usage":{"input_tokens":10,"cache_read_input_tokens":2,'
                '"cache_creation_input_tokens":null,"output_tokens":3}}'
            ),
        ]
    )

    parsed = ClaudeCodeSessionDriver._parse_stdout_jsonl(stdout)

    assert parsed.session_id == "session-123"
    assert parsed.summary == "Mutation finished."
    assert parsed.agent_turns == 2
    assert parsed.usage == {
        "input_tokens": 10,
        "cache_read_tokens": 2,
        "cache_creation_tokens": 0,
        "output_tokens": 3,
    }


def test_claude_code_spawn_collects_diff_and_transcript(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree, initial_contents="x = 1\n")
    (worktree / "AGENTS.md").write_text("# Workspace rules\n", encoding="utf-8")
    candidate = worktree / "candidate.py"
    commands: list[list[str]] = []
    operator_config_dir = tmp_path / "operator-claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(operator_config_dir))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "max")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("EVE_TEST_SENTINEL", "kept")

    def fake_run(self, *, command, cwd, env, stdout_live_path):  # noqa: ANN001
        commands.append(command)
        assert cwd == worktree
        assert command[:3] == ["claude", "-p", "Do the task"]
        assert "--resume" not in command
        assert env["CLAUDE_CONFIG_DIR"] == str(operator_config_dir)
        assert "ANTHROPIC_API_KEY" not in env
        assert "CLAUDE_CODE_EFFORT_LEVEL" not in env
        assert "CLAUDECODE" not in env
        assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
        assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "declared-token"
        assert env["EVE_TEST_SENTINEL"] == "kept"
        settings_path = worktree / ".claude" / "settings.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        excludes = settings["claudeMdExcludes"]
        assert str(operator_config_dir / "CLAUDE.md") in excludes
        assert str(worktree.resolve().parent / "AGENTS.md") in excludes
        assert settings["autoMemoryEnabled"] is False
        assert (worktree / "CLAUDE.md").read_text(encoding="utf-8") == "@AGENTS.md\n"

        stdout = _stream(session_id="session-abc", text="mutation finished", total_cost_usd=0.5)
        stdout_live_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_live_path.write_text(stdout, encoding="utf-8")
        candidate.write_text("x = 2\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(ClaudeCodeSessionDriver, "_run_command", fake_run)

    driver = ClaudeCodeSessionDriver(
        run_root=tmp_path / "run-root",
        pricing_table=_PRICING_TABLE,
        provider_env={"CLAUDE_CODE_OAUTH_TOKEN": "declared-token"},
    )
    rollout = driver.spawn(
        SessionSeed(
            instruction="Do the task",
            workspace=_workspace_lease(worktree),
        )
    )

    assert commands
    assert rollout.summary == "mutation finished"
    assert rollout.changed_paths == ["candidate.py"]
    assert rollout.usage is not None
    assert rollout.usage.input_tokens == 10
    assert rollout.usage.cache_read_tokens == 2
    assert rollout.usage.cache_creation_tokens == 4
    assert rollout.usage.output_tokens == 3
    assert rollout.usage.agent_turns == 1
    assert rollout.usage.model_cost_usd == pytest.approx(0.5)
    assert rollout.state.session_id == "session-abc"
    assert Path(rollout.state.metadata["provider_transcript_path"]).exists()


def test_claude_code_resume_reuses_session_and_transcript(
    monkeypatch,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree, initial_contents="x = 1\n")
    candidate = worktree / "candidate.py"
    launches: list[list[str]] = []

    def fake_run(self, *, command, cwd, env, stdout_live_path):  # noqa: ANN001
        launches.append(command)
        text = "spawn finished" if len(launches) == 1 else "repair finished"
        stdout = _stream(session_id="session-resume", text=text)
        candidate.write_text(f"x = {len(launches) + 1}\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(ClaudeCodeSessionDriver, "_run_command", fake_run)

    driver = ClaudeCodeSessionDriver(
        run_root=tmp_path / "run-root",
        pricing_table=_PRICING_TABLE,
        provider_env={"CLAUDE_CODE_OAUTH_TOKEN": "declared-token"},
    )
    rollout = driver.spawn(
        SessionSeed(instruction="Do the task", workspace=_workspace_lease(worktree))
    )
    resumed = driver.resume(rollout.state, instruction="Repair the boundary issue")

    assert launches[1][:3] == ["claude", "-p", "Repair the boundary issue"]
    assert launches[1][launches[1].index("--resume") + 1] == "session-resume"
    assert resumed.state.session_id == rollout.state.session_id
    assert resumed.summary == "repair finished"
    transcript = Path(resumed.state.metadata["provider_transcript_live_path"])
    assert transcript == Path(rollout.state.metadata["provider_transcript_live_path"])
    records = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    assert [record["instruction"] for record in records if record["type"] == "eve.rollout"] == [
        "Do the task",
        "Repair the boundary issue",
    ]


def test_claude_code_accepts_max_turns_as_partial_result(monkeypatch, tmp_path: Path) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree)

    def fake_run(self, *, command, cwd, env, stdout_live_path):  # noqa: ANN001
        stdout = _stream(
            session_id="session-max",
            text="still working",
            subtype="error_max_turns",
            is_error=True,
            num_turns=3,
        )
        return subprocess.CompletedProcess(args=command, returncode=1, stdout=stdout, stderr="")

    monkeypatch.setattr(ClaudeCodeSessionDriver, "_run_command", fake_run)
    driver = ClaudeCodeSessionDriver(
        run_root=tmp_path / "run-root",
        provider_env={"CLAUDE_CODE_OAUTH_TOKEN": "declared-token"},
    )

    rollout = driver.spawn(SessionSeed(instruction="Do it", workspace=_workspace_lease(worktree)))

    assert rollout.summary == "still working"
    execution = rollout.state.metadata["driver_execution"]
    assert execution["result_subtype"] == "error_max_turns"
    assert execution["accepted_partial_result"] is True
    assert rollout.usage is not None
    assert rollout.usage.agent_turns == 3


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        _stream(
            session_id="session-err",
            text="boom",
            subtype="error_during_execution",
            is_error=True,
        ),
        _stream(
            session_id="session-auth",
            text="Your organization has disabled Claude subscription access for Claude Code",
            is_error=True,
        ),
    ],
)
def test_claude_code_raises_without_usable_result(
    monkeypatch,
    tmp_path: Path,
    stdout: str,
) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree)

    def fake_run(self, *, command, cwd, env, stdout_live_path):  # noqa: ANN001
        return subprocess.CompletedProcess(args=command, returncode=1, stdout=stdout, stderr="err")

    monkeypatch.setattr(ClaudeCodeSessionDriver, "_run_command", fake_run)
    driver = ClaudeCodeSessionDriver(
        run_root=tmp_path / "run-root",
        provider_env={"CLAUDE_CODE_OAUTH_TOKEN": "declared-token"},
    )

    with pytest.raises(RuntimeError, match="Claude Code failed"):
        driver.spawn(SessionSeed(instruction="Do it", workspace=_workspace_lease(worktree)))


@pytest.mark.parametrize(
    ("web_search", "enable_multi_agent", "expected"),
    [
        ("disabled", None, {"WebSearch", "WebFetch"}),
        ("live", None, set()),
        ("live", False, {"Agent", "Task"}),
    ],
)
def test_claude_code_disallowed_tools_follow_codex_options(
    tmp_path: Path,
    web_search: str,
    enable_multi_agent: bool | None,
    expected: set[str],
) -> None:
    driver = ClaudeCodeSessionDriver(
        run_root=tmp_path / "run-root",
        web_search=web_search,
        enable_multi_agent=enable_multi_agent,
    )

    argv = driver._argv(instruction="Do it", session_id=None)  # noqa: SLF001

    disallowed = set(argv[argv.index("--disallowedTools") + 1].split(","))
    assert {"SendMessage", "ListAgents"} <= disallowed
    assert disallowed & {"WebSearch", "WebFetch", "Agent", "Task"} == expected


def test_claude_code_launch_config_keeps_workspace_instructions(tmp_path: Path) -> None:
    driver = ClaudeCodeSessionDriver(run_root=tmp_path / "run-root")
    bare = tmp_path / "bare"
    bare.mkdir()
    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "AGENTS.md").write_text("# Rules\n", encoding="utf-8")
    (owned / "CLAUDE.md").write_text("# Workspace-provided\n", encoding="utf-8")

    driver._launch_config(bare)  # noqa: SLF001
    driver._launch_config(owned)  # noqa: SLF001

    assert (bare / ".claude" / "settings.json").is_file()
    assert not (bare / "CLAUDE.md").exists()
    assert (owned / "CLAUDE.md").read_text(encoding="utf-8") == "# Workspace-provided\n"


def test_claude_code_run_command_uses_only_the_given_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    driver = ClaudeCodeSessionDriver(run_root=tmp_path / "run-root", timeout_seconds=30)
    script = (
        "import json, os; "
        "print(json.dumps({'type': 'result', 'subtype': 'success', 'result': "
        "os.environ.get('ANTHROPIC_API_KEY', 'absent')}))"
    )

    completed = driver._run_command(  # noqa: SLF001
        command=[sys.executable, "-c", script],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"]},
        stdout_live_path=tmp_path / "live.jsonl",
    )

    assert json.loads(completed.stdout)["result"] == "absent"
    assert (tmp_path / "live.jsonl").read_text(encoding="utf-8") == completed.stdout


def test_claude_code_run_command_times_out(tmp_path: Path) -> None:
    driver = ClaudeCodeSessionDriver(run_root=tmp_path / "run-root", timeout_seconds=0.5)

    with pytest.raises(RuntimeError, match="Claude Code timed out"):
        driver._run_command(  # noqa: SLF001
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            env={"PATH": os.environ["PATH"]},
            stdout_live_path=tmp_path / "live.jsonl",
        )


@pytest.mark.skipif(
    not _LIVE_CLAUDE_CODE,
    reason="Set SCALING_EVOLVE_RUN_LIVE_CLAUDE_CODE_TESTS=1 to run live Claude Code tests.",
)
def test_claude_code_live_spawn_and_resume(tmp_path: Path) -> None:
    worktree = tmp_path / "repo"
    _init_git_repo(worktree)
    driver = ClaudeCodeSessionDriver(
        run_root=tmp_path / "run-root",
        rollout_max_turns=8,
        timeout_seconds=300,
        provider_env=_LIVE_PROVIDER_ENV,
    )

    rollout = driver.spawn(
        SessionSeed(
            instruction="Change VALUE in candidate.py from 1 to 2. Remember the word heron.",
            workspace=_workspace_lease(worktree),
        )
    )
    resumed = driver.resume(
        rollout.state, instruction="Reply with the word you were asked to remember."
    )

    assert (worktree / "candidate.py").read_text(encoding="utf-8").strip() == "VALUE = 2"
    assert resumed.state.session_id == rollout.state.session_id
    assert "heron" in (resumed.summary or "").lower()
