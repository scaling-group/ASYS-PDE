"""Non-interactive Claude Code CLI-backed session driver."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from scaling_evolve.core.engine import RuntimeStateRef
from scaling_evolve.core.mutation import ProviderUsage
from scaling_evolve.providers.agent.codex_hooks import codex_hooks_payload
from scaling_evolve.providers.agent.drivers._metadata import (
    TokenPricing,
    build_driver_execution_metadata,
    compute_cost,
    resolve_token_pricing,
)
from scaling_evolve.providers.agent.drivers._subprocess import run_with_live_log
from scaling_evolve.providers.agent.drivers._transcript import archive_transcript
from scaling_evolve.providers.agent.drivers._workspace import (
    changed_paths_from_tree,
    diff_patch_from_tree,
    read_workspace_tree,
)
from scaling_evolve.providers.agent.drivers.base import (
    SessionDriver,
    SessionDriverCapabilities,
    SessionRollout,
    SessionSeed,
    SessionSnapshot,
)
from scaling_evolve.providers.agent.runtime_env import prepend_agent_runtime_bins

_LOGGER = logging.getLogger(__name__)
_TRANSCRIPT_DIRNAME = ".claude-driver-transcripts"
_WORKSPACE_EXCLUDE_DIRS = {
    ".git",
    _TRANSCRIPT_DIRNAME,
    ".codex-driver-home",
    ".codex-driver-transcripts",
}
# Inherited Anthropic/Claude Code variables are dropped from the child environment: in
# `-p` mode an ambient `ANTHROPIC_API_KEY` always wins over the subscription login,
# variables such as `CLAUDE_CODE_EFFORT_LEVEL` or `CLAUDE_CODE_USE_BEDROCK` would
# silently override the driver config, and `CLAUDECODE` marks a nested session.
# `CLAUDE_CONFIG_DIR` is kept so the operator's own login (file or macOS Keychain) is
# used. Other credentials are passed only when declared through `model_providers.*.env_key`.
_STRIPPED_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_")
_STRIPPED_ENV_NAMES = frozenset({"CLAUDECODE"})
_MAX_TURNS_SUBTYPE = "error_max_turns"
# Session-orchestration tools have no use inside a rollout, and `SendMessage`/`ListAgents`
# reach the operator's other Claude sessions on the same machine.
_ORCHESTRATION_TOOLS = (
    "SendMessage",
    "ListAgents",
    "RemoteTrigger",
    "CronCreate",
    "CronDelete",
    "CronList",
    "ScheduleWakeup",
    "Workflow",
    "EnterWorktree",
    "ExitWorktree",
)
_WORKSPACE_INSTRUCTIONS = "@AGENTS.md\n"
_SYNTHETIC_MODEL = "<synthetic>"


@dataclass(frozen=True)
class ClaudeCodeStreamSummary:
    """Parsed fields from one `claude -p --output-format stream-json` stdout."""

    session_id: str | None
    summary: str | None
    result: dict[str, object] | None
    usage: dict[str, int]
    agent_turns: int


class ClaudeCodeSessionDriver(SessionDriver):
    """Spawn-and-resume Claude Code driver backed by `claude -p --output-format stream-json`."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        executable: str = "claude",
        model: str = "claude-sonnet-5-5",
        reasoning_effort: str = "low",
        rollout_max_turns: int = 200,
        budget_prompt: bool = True,
        enable_multi_agent: bool | None = None,
        timeout_seconds: float = 900.0,
        system_prompt_file: str | None = None,
        role: str | None = None,
        web_search: str = "disabled",
        token_pricing: TokenPricing | None = None,
        pricing_table: Mapping[str, TokenPricing] | None = None,
        provider_env: dict[str, str] | None = None,
    ) -> None:
        self.run_root = Path(run_root).expanduser().resolve()
        self.executable = executable
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.rollout_max_turns = rollout_max_turns
        self.budget_prompt = budget_prompt
        self.enable_multi_agent = enable_multi_agent
        self.timeout_seconds = timeout_seconds
        # Replaces Claude Code's built-in system prompt; None keeps the default.
        self.system_prompt_file = (
            Path(system_prompt_file).expanduser().resolve()
            if system_prompt_file is not None
            else None
        )
        self.role = role
        self.web_search = web_search if web_search in {"disabled", "live", "cached"} else "disabled"
        self.token_pricing = resolve_token_pricing(model, token_pricing, pricing_table)
        self.provider_env = dict(provider_env or {})

    def capabilities(self) -> SessionDriverCapabilities:
        return SessionDriverCapabilities(
            supports_native_fork=False,
            supports_cross_workspace_fork=False,
        )

    def workspace_config_dir(self, worktree_root: Path) -> Path:
        """Return the workspace's project-scope `.claude` directory."""

        return worktree_root / ".claude"

    def spawn(self, seed: SessionSeed) -> SessionRollout:
        workspace = seed.workspace
        if workspace is None:
            raise ValueError("ClaudeCodeSessionDriver requires a resolved workspace lease.")
        worktree_root = Path(workspace.session_cwd).resolve()
        return self._run_rollout(
            instruction=seed.instruction,
            worktree_root=worktree_root,
            workspace_id=workspace.workspace_id,
            target_repo_root=workspace.target_repo_root,
            workspace_root=workspace.workspace_root,
            session_cwd=workspace.session_cwd,
            session_id=None,
            state_id=f"runtime:{uuid4().hex}",
            metadata={
                **dict(seed.display_context),
                "prompt_file": seed.prompt_file,
                "write_prompt_file": seed.write_prompt_file,
            },
        )

    def fork_session(self, parent: RuntimeStateRef) -> str:
        _ = parent
        raise NotImplementedError("claude_code is spawn-only; native fork is unsupported.")

    def migrate_session(self, *, parent_cwd: str, child_cwd: str, session_id: str) -> str:
        _ = (parent_cwd, child_cwd, session_id)
        raise NotImplementedError("claude_code is spawn-only; migrate_session is unsupported.")

    def fork(self, parent: RuntimeStateRef, instruction: str) -> SessionRollout:
        _ = (parent, instruction)
        raise NotImplementedError("claude_code is spawn-only; fork is unsupported.")

    def resume(self, state: RuntimeStateRef, instruction: str | None = None) -> SessionRollout:
        if not state.session_id:
            raise ValueError("claude_code resume requires a session_id.")

        worktree_root = Path(state.session_cwd or state.workspace_root or ".").resolve()
        return self._run_rollout(
            instruction=instruction or "continue",
            worktree_root=worktree_root,
            workspace_id=state.workspace_id,
            target_repo_root=state.target_repo_root,
            workspace_root=state.workspace_root,
            session_cwd=str(worktree_root),
            session_id=state.session_id,
            state_id=state.state_id,
            metadata=dict(state.metadata),
        )

    def snapshot(self, state: RuntimeStateRef) -> SessionSnapshot:
        _ = state
        raise NotImplementedError("claude_code is spawn-only; snapshot is unsupported.")

    def _run_rollout(
        self,
        *,
        instruction: str,
        worktree_root: Path,
        workspace_id: str | None,
        target_repo_root: str | None,
        workspace_root: str | None,
        session_cwd: str | None,
        session_id: str | None,
        state_id: str,
        metadata: dict[str, object],
    ) -> SessionRollout:
        prompt_file = _string_metadata(metadata.get("prompt_file"))
        write_prompt_file = bool(metadata.get("write_prompt_file", True))
        instruction_path: Path | None = None
        if prompt_file is not None:
            instruction_path = worktree_root / prompt_file
        if write_prompt_file and instruction_path is not None:
            instruction_path.write_text(instruction.strip() + "\n", encoding="utf-8")
        elif instruction_path is not None and not instruction_path.exists():
            raise FileNotFoundError(f"Prompt file does not exist: {instruction_path}")

        self._launch_config(worktree_root)
        before_tree = read_workspace_tree(worktree_root, exclude_dirs=_WORKSPACE_EXCLUDE_DIRS)
        initial_head = _git_head(worktree_root)
        launch_started_ns = time.time_ns()
        argv = self._argv(instruction=instruction, session_id=session_id)
        transcript_root = self._transcript_root(worktree_root)
        transcript_root.mkdir(parents=True, exist_ok=True)
        driver_stdout_live_path = self._driver_stdout_live_path(worktree_root)
        completed = self._run_command(
            command=argv,
            cwd=worktree_root,
            env=self._rollout_env(worktree_root),
            stdout_live_path=driver_stdout_live_path,
        )

        stream_summary = self._parse_stdout_jsonl(completed.stdout)
        result = stream_summary.result
        result_subtype = _string(result.get("subtype")) if result is not None else None
        if result is None or _result_is_error(result):
            raise RuntimeError(
                self._format_execution_failure(
                    command=argv,
                    cwd=worktree_root,
                    completed=completed,
                    result_subtype=result_subtype,
                )
            )
        if completed.returncode != 0:
            _LOGGER.warning(
                "Claude Code exited with code %s but returned usable subtype `%s` in %s",
                completed.returncode,
                result_subtype or "unknown",
                worktree_root,
            )
        rollout_max_turns_reached = result_subtype == _MAX_TURNS_SUBTYPE

        final_head = _git_head(worktree_root)
        after_tree = read_workspace_tree(worktree_root, exclude_dirs=_WORKSPACE_EXCLUDE_DIRS)
        changed_paths = changed_paths_from_tree(before_tree, after_tree)
        attempt_label = _attempt_label(prefix="resume" if session_id else "spawn")
        patch_path = transcript_root / f"{attempt_label}-diff.patch"
        patch_path.write_text(diff_patch_from_tree(before_tree, after_tree), encoding="utf-8")

        resolved_session_id = stream_summary.session_id or session_id or f"session:{uuid4().hex}"
        live_session_path = self._cumulative_transcript_path(
            worktree_root,
            session_id=resolved_session_id,
            previous_path=(
                _string_metadata(metadata.get("provider_transcript_live_path"))
                if session_id is not None
                else None
            ),
        )
        self._append_cumulative_transcript(
            live_session_path,
            stdout=completed.stdout,
            instruction=instruction,
            session_id=resolved_session_id,
            cwd=worktree_root,
            truncate=session_id is None,
        )
        session_archive_path = archive_transcript(
            live_session_path,
            transcript_root,
            _safe_name(resolved_session_id),
            timestamp_ns=time.time_ns(),
        )

        summary = stream_summary.summary
        if rollout_max_turns_reached and summary is None:
            summary = _rollout_max_turns_summary(stream_summary.agent_turns)
        completion_payload = {
            "status": "ok",
            "summary": summary,
            "changed_files": changed_paths,
        }
        completion_path = transcript_root / f"{attempt_label}-completion.json"
        completion_path.write_text(
            json.dumps(completion_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for stale_key in (
            "attempt_root",
            "instruction_path",
            "completion_path",
            "diff_path",
            "provider_transcript_path",
            "provider_transcript_live_path",
            "driver_stdout_live_path",
            "driver_stdout",
            "driver_stderr",
            "initial_head",
            "final_head",
            "actual_execution_mode",
            "driver_execution",
        ):
            metadata.pop(stale_key, None)
        metadata.update(
            {
                "driver": "claude_code",
                "role": self.role,
                "attempt_root": str(transcript_root),
                "instruction_path": str(instruction_path) if instruction_path is not None else None,
                "completion_path": str(completion_path),
                "diff_path": str(patch_path),
                "provider_transcript_path": str(session_archive_path),
                "provider_transcript_live_path": str(live_session_path),
                "driver_stdout_live_path": str(driver_stdout_live_path),
                "driver_stdout": completed.stdout,
                "driver_stderr": completed.stderr,
                "initial_head": initial_head,
                "final_head": final_head,
                "actual_execution_mode": "claude_code",
                "driver_execution": build_driver_execution_metadata(
                    driver="claude_code",
                    command=argv,
                    cwd=worktree_root,
                    exit_code=completed.returncode,
                    rollout_max_turns=self.rollout_max_turns,
                    timeout_seconds=self.timeout_seconds,
                    model=self.model,
                    effort_level=self.reasoning_effort,
                    reasoning_effort=self.reasoning_effort,
                    result_subtype=result_subtype,
                    result_is_error=False,
                    accepted_partial_result=rollout_max_turns_reached,
                    num_turns=stream_summary.agent_turns,
                    web_search=self.web_search,
                ),
                "prompt_file": prompt_file,
                "write_prompt_file": write_prompt_file,
            }
        )

        parsed_usage = ProviderUsage(
            input_tokens=stream_summary.usage["input_tokens"],
            output_tokens=stream_summary.usage["output_tokens"],
            cache_read_tokens=stream_summary.usage["cache_read_tokens"],
            cache_creation_tokens=stream_summary.usage["cache_creation_tokens"],
            # Claude Code's own estimate; `compute_cost` falls back to pricing without it.
            model_cost_usd=_float_value(result.get("total_cost_usd")),
            wallclock_seconds=(time.time_ns() - launch_started_ns) / 1_000_000_000,
            agent_turns=stream_summary.agent_turns,
        )
        usage = parsed_usage.model_copy(
            update={"model_cost_usd": compute_cost(parsed_usage, self.token_pricing)}
        )
        state = RuntimeStateRef(
            state_id=state_id,
            provider_kind="claude_code",
            session_id=resolved_session_id,
            workspace_id=workspace_id,
            target_repo_root=target_repo_root,
            workspace_root=workspace_root,
            session_cwd=session_cwd or str(worktree_root),
            metadata={key: value for key, value in metadata.items() if value is not None},
        )
        return SessionRollout(
            state=state,
            primary_path=changed_paths[0] if changed_paths else None,
            changed_paths=changed_paths,
            summary=summary,
            usage=usage,
        )

    def _argv(self, *, instruction: str, session_id: str | None) -> list[str]:
        prompt = instruction.strip()
        if prompt.startswith("-"):
            # Keep a prompt that begins with a dash from being parsed as a CLI option.
            prompt = f" {prompt}"
        # Never pass `--bare`: it skips hooks and only authenticates with API keys.
        command = [
            self.executable,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "bypassPermissions",
            # Workspace settings only: the operator's personal settings stay out.
            "--setting-sources",
            "project,local",
            "--max-turns",
            str(self.rollout_max_turns),
        ]
        if session_id is not None:
            command.extend(["--resume", session_id])
        if self.model:
            command.extend(["--model", self.model])
        if self.reasoning_effort:
            command.extend(["--effort", self.reasoning_effort])
        if self.system_prompt_file is not None:
            if not self.system_prompt_file.is_file():
                raise FileNotFoundError(
                    f"Claude Code system prompt file does not exist: {self.system_prompt_file}"
                )
            command.extend(["--system-prompt-file", str(self.system_prompt_file)])
        disallowed_tools = self._disallowed_tools()
        if disallowed_tools:
            # Variadic flag: keep it last so it cannot swallow other arguments.
            command.extend(["--disallowedTools", ",".join(disallowed_tools)])
        return command

    def _disallowed_tools(self) -> list[str]:
        tools: list[str] = list(_ORCHESTRATION_TOOLS)
        if self.web_search == "disabled":
            tools.extend(["WebSearch", "WebFetch"])
        if self.enable_multi_agent is False:
            # Claude Code releases name the subagent tool `Agent` or `Task`.
            tools.extend(["Agent", "Task"])
        return tools

    def _launch_config(self, worktree_root: Path) -> None:
        """Write the workspace's Claude Code settings and instructions for one launch."""

        config_dir = self.workspace_config_dir(worktree_root)
        config_dir.mkdir(parents=True, exist_ok=True)
        settings = {
            # Codex adopted Claude Code's hook schema, so the shared payload applies as is.
            "hooks": codex_hooks_payload()["hooks"],
            # Treat the workspace as the project boundary: personal memory and memory
            # files of enclosing directories must not leak into rollout agents.
            "claudeMdExcludes": _outside_memory_excludes(worktree_root),
            "autoMemoryEnabled": False,
        }
        (config_dir / "settings.json").write_text(
            json.dumps(settings, indent=2) + "\n",
            encoding="utf-8",
        )
        instructions = worktree_root / "CLAUDE.md"
        if (worktree_root / "AGENTS.md").is_file() and not instructions.exists():
            instructions.write_text(_WORKSPACE_INSTRUCTIONS, encoding="utf-8")

    def _rollout_env(self, worktree_root: Path) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in _STRIPPED_ENV_NAMES and not key.startswith(_STRIPPED_ENV_PREFIXES)
        }
        env.update(self.provider_env)
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        # `claude -p` waits for background shells and subagents before it exits, so a
        # backgrounded auditor would hold the rollout until the timeout.
        env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
        return prepend_agent_runtime_bins(env, workspace_root=worktree_root)

    def _run_command(
        self,
        *,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        stdout_live_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        try:
            # `env` is the complete child environment so the stripped Anthropic/Claude
            # variables stay removed.
            return run_with_live_log(
                command, cwd, env, stdout_live_path, self.timeout_seconds, inherit_env=False
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                self._format_timeout_failure(
                    command=command,
                    cwd=cwd,
                    stdout=_timeout_text(error.output),
                    stderr=_timeout_text(error.stderr),
                )
            ) from error

    @staticmethod
    def _parse_stdout_jsonl(stdout: str) -> ClaudeCodeStreamSummary:
        session_id: str | None = None
        result: dict[str, object] | None = None
        last_text: str | None = None
        main_message_ids: set[str] = set()
        message_usage: dict[str, dict[str, object]] = {}
        for line_number, line in enumerate(stdout.splitlines()):
            payload = _load_json_line(line.strip())
            if payload is None:
                continue
            session_id = _string(payload.get("session_id")) or session_id
            payload_type = _string(payload.get("type"))
            if payload_type == "assistant":
                message = _mapping(payload.get("message"))
                message_id = _string(message.get("id")) or f"line:{line_number}"
                if isinstance(message.get("usage"), dict):
                    message_usage[message_id] = _mapping(message.get("usage"))
                if payload.get("parent_tool_use_id"):
                    # Subagent traffic interleaved into the stream; count the main thread only.
                    continue
                main_message_ids.add(message_id)
                text = _message_text(message).strip()
                if text and not _is_cli_notice(payload):
                    last_text = text
            elif payload_type == "result":
                result = payload

        if result is not None and isinstance(result.get("usage"), dict):
            usage = _usage_from_payload(_mapping(result.get("usage")))
        else:
            # A killed run has no result event; fall back to per-message usage.
            usage = _empty_usage()
            for message_payload in message_usage.values():
                for key, value in _usage_from_payload(message_payload).items():
                    usage[key] += value
        num_turns = result.get("num_turns") if result is not None else None
        agent_turns = (
            num_turns
            if isinstance(num_turns, int) and not isinstance(num_turns, bool)
            else len(main_message_ids)
        )
        result_text = _string(result.get("result")) if result is not None else None
        summary = (
            result_text.strip()
            if result_text and result_text.strip() and not _result_is_error(result or {})
            else last_text
        )
        return ClaudeCodeStreamSummary(
            session_id=session_id,
            summary=summary,
            result=result,
            usage=usage,
            agent_turns=agent_turns,
        )

    def _append_cumulative_transcript(
        self,
        path: Path,
        *,
        stdout: str,
        instruction: str,
        session_id: str,
        cwd: Path,
        truncate: bool,
    ) -> None:
        context = {
            "type": "eve.rollout",
            "timestamp": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
            "session_id": session_id,
            "instruction": instruction,
            "model": self.model,
            "effort": self.reasoning_effort,
            "role": self.role,
            "cwd": str(cwd),
        }
        with path.open("w" if truncate else "a", encoding="utf-8") as handle:
            handle.write(json.dumps(context, ensure_ascii=False, sort_keys=True) + "\n")
            handle.write(stdout)
            if stdout and not stdout.endswith("\n"):
                handle.write("\n")

    def _transcript_root(self, cwd: Path) -> Path:
        return cwd / _TRANSCRIPT_DIRNAME

    def _driver_stdout_live_path(self, cwd: Path) -> Path:
        filename = f"attempt-{time.time_ns()}-{uuid4().hex[:8]}-live.jsonl"
        return self._transcript_root(cwd) / filename

    def _cumulative_transcript_path(
        self,
        cwd: Path,
        *,
        session_id: str,
        previous_path: str | None,
    ) -> Path:
        if previous_path is not None:
            candidate = Path(previous_path).expanduser()
            if candidate.is_file():
                return candidate
        return self._transcript_root(cwd) / f"session-{_safe_name(session_id)}.jsonl"

    def _format_execution_failure(
        self,
        *,
        command: list[str],
        cwd: Path,
        completed: subprocess.CompletedProcess[str],
        result_subtype: str | None,
    ) -> str:
        reason = result_subtype or "missing result event"
        return "\n".join(
            [
                f"Claude Code failed ({reason}).",
                f"cwd: {cwd}",
                f"exit_code: {completed.returncode}",
                f"command: {_command_for_log(command)}",
                f"stdout:\n{completed.stdout}",
                f"stderr:\n{completed.stderr}",
            ]
        )

    def _format_timeout_failure(
        self,
        *,
        command: list[str],
        cwd: Path,
        stdout: str,
        stderr: str,
    ) -> str:
        return "\n".join(
            [
                "Claude Code timed out.",
                f"cwd: {cwd}",
                f"timeout_seconds: {self.timeout_seconds}",
                f"command: {_command_for_log(command)}",
                f"stdout:\n{stdout}",
                f"stderr:\n{stderr}",
            ]
        )


def _outside_memory_excludes(worktree_root: Path) -> list[str]:
    user_config_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    patterns = [str(user_config_dir.expanduser() / "CLAUDE.md")]
    for ancestor in worktree_root.resolve().parents:
        patterns.extend(
            [
                str(ancestor / "CLAUDE.md"),
                str(ancestor / "CLAUDE.local.md"),
                str(ancestor / "AGENTS.md"),
                str(ancestor / ".claude" / "**"),
            ]
        )
    return patterns


def _result_is_error(result: dict[str, object]) -> bool:
    subtype = _string(result.get("subtype"))
    if subtype == _MAX_TURNS_SUBTYPE:
        return False
    # Authentication and API failures arrive as `subtype: success` with `is_error: true`.
    is_error = result.get("is_error")
    if isinstance(is_error, bool):
        return is_error
    return subtype != "success"


def _usage_from_payload(payload: dict[str, object]) -> dict[str, int]:
    return {
        "input_tokens": _int_value(payload.get("input_tokens")),
        "cache_read_tokens": _int_value(payload.get("cache_read_input_tokens")),
        "cache_creation_tokens": _int_value(payload.get("cache_creation_input_tokens")),
        "output_tokens": _int_value(payload.get("output_tokens")),
    }


def _empty_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "output_tokens": 0,
    }


def _is_cli_notice(payload: dict[str, object]) -> bool:
    """Return whether an assistant event is a CLI-synthesized error or notice."""

    message = _mapping(payload.get("message"))
    return bool(payload.get("error")) or message.get("model") == _SYNTHETIC_MODEL


def _message_text(message: dict[str, object]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        block.get("text")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in parts if isinstance(part, str))


def _command_for_log(command: list[str]) -> list[str]:
    """Shorten the prompt argument so failures stay readable."""

    rendered = list(command)
    if len(rendered) > 2 and rendered[1] == "-p" and len(rendered[2]) > 200:
        rendered[2] = rendered[2][:200] + "...[truncated]"
    return rendered


def _attempt_label(*, prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"


def _git_head(cwd: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _load_json_line(line: str) -> dict[str, object] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_metadata(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _int_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


def _float_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _rollout_max_turns_summary(turns: int) -> str:
    if turns > 0:
        return f"Claude Code reached rollout_max_turns after {turns} turns."
    return "Claude Code reached rollout_max_turns before returning a final summary."


def _timeout_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""
