from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import scaling_evolve.algorithms.eve.runner  # noqa: F401
from scaling_evolve.algorithms.eve.runtime.driver import build_driver, build_role_drivers
from scaling_evolve.providers.agent.drivers.claude_code import ClaudeCodeSessionDriver


def _compose_with_driver(driver: str, *, config_name: str = "circle_packing.smoke"):
    config_dir = Path("configs/eve").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name=config_name,
            overrides=[f"driver={driver}"],
        )


def test_prompt_overrides_build_claude_role_drivers(tmp_path: Path) -> None:
    cfg = _compose_with_driver("claude_smoke", config_name="circle_packing.smoke")
    OmegaConf.set_struct(cfg.driver, False)
    cfg.driver.overrides = {
        "solver": {
            "system_prompt_file": (
                "configs/eve/optimizer/circle_packing/prompt/CODEX_SYSTEM_PROMPT.md"
            )
        },
        "eval": {
            "system_prompt_file": (
                "configs/eve/evaluation/circle_packing/prompt/CODEX_SYSTEM_PROMPT.md"
            )
        },
    }
    driver_cfg = OmegaConf.to_container(cfg.driver, resolve=True)
    assert isinstance(driver_cfg, dict)

    drivers = build_role_drivers(driver_cfg, run_root=tmp_path, worker_slots=1)

    solver = drivers.solver_driver
    judge = drivers.eval_driver_factory()
    assert isinstance(solver, ClaudeCodeSessionDriver)
    assert isinstance(judge, ClaudeCodeSessionDriver)
    assert solver.system_prompt_file != judge.system_prompt_file


def test_claude_code_reuses_declared_provider_env_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EVE_TEST_CLAUDE_TOKEN", "declared-token")

    driver = build_driver(
        {
            "driver": "claude_code",
            "effort_level": "high",
            "model_providers": {"claude": {"env_key": "EVE_TEST_CLAUDE_TOKEN"}},
        },
        run_root=tmp_path,
    )

    assert isinstance(driver, ClaudeCodeSessionDriver)
    assert driver.reasoning_effort == "high"
    assert driver.provider_env == {"EVE_TEST_CLAUDE_TOKEN": "declared-token"}


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"personality": "pragmatic"}, "does not support Codex-specific options"),
        ({"sandbox_mode": "workspace-write"}, "does not support Codex-specific options"),
        ({"variant": "high"}, "only supported by opencode"),
        (
            {"model_providers": {"claude": {"base_url": "https://example.invalid"}}},
            "only reuses `model_providers.*.env_key`",
        ),
    ],
)
def test_claude_code_rejects_options_it_cannot_honor(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(SystemExit, match=message):
        build_driver({"driver": "claude_code", **override}, run_root=tmp_path)
