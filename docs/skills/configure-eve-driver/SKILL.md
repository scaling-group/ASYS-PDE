---
name: configure-eve-driver
description: "Use when choosing, editing, or overriding EvE driver settings for Codex, OpenCode, Claude Code, or interactive/debug runs."
---

# Configure the EvE Driver

Use the public presets first:

- `driver: codex_smoke` is for fast runtime validation.
- `driver: codex_max` is for full non-interactive runs.
- `driver: opencode_smoke` is the equivalent fast OpenCode path.
- `driver: opencode_max` is the equivalent full-run OpenCode path.
- `driver: claude_smoke` and `driver: claude_max` are the equivalent Claude Code paths.

Do not add a new preset for a one-off variation. Prefer Hydra overrides until the role is stable enough to document as a supported entrypoint.

## Public Surface

Driver presets live in `configs/eve/driver/`. Keep that directory small:

- `codex_smoke.yaml`: non-interactive Codex backend, small turn budget, web search disabled.
- `codex_max.yaml`: non-interactive Codex backend, full-run budget, web search enabled.
- `opencode_smoke.yaml`: non-interactive OpenCode backend with a small native step cap.
- `opencode_max.yaml`: non-interactive OpenCode backend with a full-run native step cap.
- `claude_smoke.yaml`: non-interactive Claude Code backend, low effort, small native turn cap, web tools disabled.
- `claude_max.yaml`: non-interactive Claude Code backend, full-run turn cap, web tools enabled.

Task smoke configs live beside task configs, but keep the root config surface compact:

- `circle_packing.smoke` is the single tracked circle packing smoke preset.
- By default it validates the ordinary loop.
- Use explicit Hydra overrides from that same entrypoint for judge evaluation or role-specific system prompt smoke paths.

Task smoke configs use `driver: codex_smoke` by default; select `driver=opencode_smoke` or `driver=claude_smoke` to exercise OpenCode or Claude Code. Do not shrink `loop.n_workers_phase2` in the default smoke path; the two-worker default covers Phase 2 multi-worker behavior and optimizer sync. Use `loop.n_parallel_phase2` when you only need to reduce concurrent solver agents.

## Normal Commands

Fast smoke:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=circle_packing.smoke
```

OpenCode smoke (use a native `provider/model` identifier):

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=circle_packing.smoke driver=opencode_smoke driver.model=deepseek/deepseek-chat
```

Judge evaluation smoke:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner \
  --config-name=circle_packing.smoke \
  evaluation=circle_packing.judge \
  label=circle-packing-judge-smoke
```

Role-specific system prompt smoke:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner \
  --config-name=circle_packing.smoke \
  evaluation=circle_packing.judge \
  loop.max_iterations=1 \
  label=circle-packing-codex-prompt-smoke \
  +driver.overrides.solver.system_prompt_file=configs/eve/optimizer/circle_packing/prompt/CODEX_SYSTEM_PROMPT.md
```

Full run:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=circle_packing
```

OpenCode full run:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=circle_packing driver=opencode_max driver.model=deepseek/deepseek-chat
```

Claude Code smoke and full run:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=circle_packing.smoke driver=claude_smoke
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=circle_packing driver=claude_max
```

One-off interactive/debug run:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner \
  --config-name=circle_packing.smoke \
  driver.driver=codex_tmux \
  driver.open_iterm2=true
```

If overriding from a full config, keep the backend switch explicit:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner \
  --config-name=circle_packing \
  driver.driver=codex_tmux \
  driver.open_iterm2=true
```

Do not preserve interactive/debug overrides as new YAML presets unless the team decides they are a supported public entry.

## Selection Semantics

The driver config lives under the top-level `driver:` key.

- `driver.driver` selects the backend and wins over `driver.provider`.
- `driver.provider` is accepted for older configs. Avoid it for new public presets.
- If a config inherits from `codex_max`, changing only `driver.provider` will not switch the backend because `driver.driver=codex_exec` is still set.
- Role overrides are shallow merges: EvE starts from the base `driver` mapping, then overlays `driver.overrides.solver` or `driver.overrides.eval`.
- Do not configure `driver.overrides.optimizer`; the independent optimizer driver role was removed.

Supported backend names are `codex_exec`, `codex_tmux`, `opencode`, and `claude_code`.

## Authentication

- Codex uses its native authentication by default; alternate providers may use `model_providers.*.env_key`.
- OpenCode uses credentials saved by `opencode auth login` unless an explicit `model_providers.*.env_key` is configured.
- Claude Code uses the operator's own login from `claude` `/login` (a credentials file, or the Keychain on macOS; `CLAUDE_CONFIG_DIR` is honored) unless an explicit `model_providers.*.env_key` is configured, for example `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token` or `ANTHROPIC_API_KEY`. Inherited `ANTHROPIC_*`/`CLAUDE_CODE_*` variables are removed: in `-p` mode an ambient `ANTHROPIC_API_KEY` would otherwise take precedence over the subscription login.
- Explicit environment credentials take precedence. Do not rely on undeclared ambient variables.

## Driver Options

Common fields:

- `driver.executable`: command name to launch; defaults to `codex`, `opencode`, or `claude` according to the backend.
- `driver.model`: backend-native model identifier.
- `driver.rollout_max_turns`: per-rollout agent turn/step cap.
- `driver.timeout_seconds`: per-rollout wall-clock timeout.
- `driver.system_prompt_file`: repo-relative path to a backend system prompt file.
- `driver.model_providers`: provider config map; if an entry declares `env_key`, EvE fails fast when that environment variable is missing.
- `driver.token_pricing`: per-driver pricing metadata override. Prefer shared pricing in `configs/pricing.yaml` unless a run needs a local override.

Codex-style fields (`codex_exec`, `codex_tmux`, `claude_code`):

- `driver.reasoning_effort`: reasoning budget where supported.
- `driver.effort_level`: accepted as a fallback spelling for Codex-style reasoning effort.
- `driver.web_search`: `disabled`, `cached`, or `live`.
- `driver.budget_prompt`: whether EvE injects the budget prompt wrapper.
- `driver.enable_multi_agent`: optional backend feature override.

Codex-only fields (`codex_exec`, `codex_tmux`):

- `driver.personality`: optional backend personality override.
- `driver.model_provider`: optional alternate provider id for Codex-style backends.

OpenCode-only semantics (`opencode`):

- `driver.model` must use OpenCode's native `provider/model` form when set; omit it to use OpenCode's configured default.
- `driver.variant` passes the provider-specific native variant unchanged.
- `driver.rollout_max_turns` is applied through OpenCode's native `agent.build.steps` setting.
- `driver.budget_prompt` must be `false`; Codex hook injection is not translated.
- Codex-only options are rejected instead of ignored or mapped.
- HOME, XDG state, project discovery, and external skills are isolated per EvE workspace.
- EvE-owned `AGENTS.md`, guidance skills, and helper agents are projected into the isolated
  OpenCode configuration.
- Only `model_providers.*.env_key` is reused; other Codex provider settings are rejected
  instead of translated.
- Spawn creates a native OpenCode session; resume uses its exact session ID in the same workspace. Native fork, cross-workspace migration, and snapshots are unsupported.

Claude Code semantics (`claude_code`):

- Runs `claude -p <prompt> --output-format stream-json --verbose --permission-mode bypassPermissions --max-turns <rollout_max_turns>`, adding `--resume <session_id>` on resume; never `--bare`, which skips hooks.
- `driver.reasoning_effort` is passed as `--effort` (`low`, `medium`, `high`, `xhigh`, or `max`).
- `driver.rollout_max_turns` is Claude Code's native `--max-turns`; a run that reaches it (`error_max_turns`) is kept as a partial candidate.
- `driver.system_prompt_file` replaces Claude Code's built-in system prompt through `--system-prompt-file`, like Codex's `model_instructions_file`.
- `driver.web_search: disabled` disallows the `WebSearch` and `WebFetch` tools; `driver.enable_multi_agent: false` disallows the `Agent` (subagent) tool. Session-orchestration tools (`SendMessage`, `ListAgents`, `RemoteTrigger`, cron and wakeup tools, `Workflow`, worktree tools) are always disallowed, and background tasks are off (`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`) because `claude -p` would wait for them until the timeout.
- `driver.budget_prompt` and the workspace guard use the same hook payload as Codex, written to the isolated `settings.json`.
- The workspace's `.claude/` mirrors `.codex/`: EvE writes `.claude/settings.json` at launch (hooks, `claudeMdExcludes` for the operator's personal `CLAUDE.md` and memory files of enclosing directories, auto memory off), renders `.codex/agents/*.toml` helper agents to `.claude/agents/*.md`, and links `.claude/skills` to `.codex/skills`. `--setting-sources project,local` keeps the operator's personal settings out. A workspace `AGENTS.md` is loaded through a generated `CLAUDE.md` containing `@AGENTS.md`.
- Only `model_providers.*.env_key` is reused; `driver.personality`, `driver.model_provider`, `driver.variant`, and tmux-only fields are rejected.
- Cost comes from Claude Code's `total_cost_usd` estimate, with `configs/pricing.yaml` as the fallback. Native fork, cross-workspace migration, and snapshots are unsupported.
- Claude Code keeps each rollout's native session under the operator's `~/.claude/projects/`, one directory per workspace path; EvE does not remove them, so clear them after a run when they are no longer needed for resume or debugging.

Tmux-only fields (`codex_tmux`):

- `driver.approval_policy`: approval policy passed to the interactive backend.
- `driver.sandbox_mode`: sandbox mode passed to the interactive backend.
- `driver.allow_network`: when `sandbox_mode` is omitted, `true` maps to `danger-full-access`; otherwise the default is `workspace-write`.
- `driver.open_iterm2`: whether EvE opens the tmux session in a visible terminal window.
- `driver.completion_filename`: internal tmux completion marker filename.
- `driver.instruction_filename`: internal tmux instruction filename.

## Implementation References

This skill is grounded in the current runtime surfaces:

- `src/scaling_evolve/algorithms/eve/runtime/driver.py` for backend selection, role overrides, tmux pool creation, sandbox/web-search normalization, provider environment resolution, and pricing config loading.
- `src/scaling_evolve/providers/agent/drivers/codex_exec.py` for non-interactive Codex fields.
- `src/scaling_evolve/providers/agent/drivers/opencode.py` for non-interactive OpenCode spawn/resume, JSONL parsing, and native step limits.
- `src/scaling_evolve/providers/agent/drivers/claude_code.py` for non-interactive Claude Code spawn/resume, workspace settings, and stream-json parsing.
- `src/scaling_evolve/providers/agent/drivers/codex_tmux.py` for interactive Codex/tmux fields.
- `src/scaling_evolve/providers/agent/config.py` for persistent session provider config validation.
