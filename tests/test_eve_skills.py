from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


def test_eve_initial_guidance_skills_have_valid_yaml_frontmatter() -> None:
    skill_paths = (
        Path(
            "configs/eve/optimizer/circle_packing/initial_guidance/skills/"
            "optimize-from-examples/SKILL.md"
        ),
        Path("configs/eve/optimizer/circle_packing/initial_guidance/skills/read-eval/SKILL.md"),
    )

    for skill_path in skill_paths:
        content = skill_path.read_text(encoding="utf-8")
        _leading, _separator, remainder = content.partition("---\n")
        frontmatter_block, _separator, _body = remainder.partition("---\n")
        payload = yaml.safe_load(frontmatter_block)
        assert payload["name"]
        assert payload["description"]


def test_repo_docs_skills_have_valid_yaml_frontmatter() -> None:
    skill_paths = sorted(Path("docs/skills").glob("*/SKILL.md"))

    for skill_path in skill_paths:
        content = skill_path.read_text(encoding="utf-8")
        _leading, _separator, remainder = content.partition("---\n")
        frontmatter_block, _separator, _body = remainder.partition("---\n")
        payload = yaml.safe_load(frontmatter_block)
        assert payload["name"]
        assert payload["description"]

    assert skill_paths


def test_eve_codex_check_agents_have_valid_toml() -> None:
    agent_paths = (
        (
            Path("configs/eve/optimizer/circle_packing/immutable/.codex/agents/check-runner.toml"),
            "check-runner",
        ),
        (
            Path(
                "configs/eve/evaluation/circle_packing/immutable_assess/"
                ".codex/agents/score-check.toml"
            ),
            "score-check",
        ),
    )

    for agent_path, expected_name in agent_paths:
        payload = tomllib.loads(agent_path.read_text(encoding="utf-8"))
        assert payload["name"] == expected_name
        assert payload["description"]
        assert payload["developer_instructions"]


def test_circle_packing_smoke_uses_single_root_preset_with_override_recipes() -> None:
    smoke_path = Path("configs/eve/circle_packing.smoke.yaml")
    smoke_text = smoke_path.read_text(encoding="utf-8")

    assert smoke_path.is_file()
    assert not Path("configs/eve/circle_packing.codex_prompt.smoke.yaml").exists()
    assert not Path("configs/eve/circle_packing.judge.smoke.yaml").exists()
    assert yaml.safe_load(smoke_text)["loop"]["max_iterations"] == 2
    assert "evaluation=circle_packing.judge" in smoke_text
    assert "+driver.overrides.solver.system_prompt_file" in smoke_text
    assert "+driver.overrides.eval.system_prompt_file" in smoke_text
