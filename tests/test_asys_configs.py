"""ASYS config compatibility with the current EvE interfaces."""

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import scaling_evolve.algorithms.eve.runner  # noqa: F401  (registers runtime resolvers)
from scaling_evolve.algorithms.eve.factory import _load_solver_worker_configs
from scaling_evolve.algorithms.eve.problem.repo import RepoTaskProblem
from scaling_evolve.algorithms.eve.runtime.driver import _driver_cfg_for_role
from scaling_evolve.algorithms.eve.workflow.evaluation import _validate_score_shape

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/eve"
CASES = (
    "asys-gclm-profile",
    "asys-fujita-ivp",
    "nonlinear-schrodinger",
    "asys-pinn-allen-cahn-2d",
    "keller-segel-radial-2d",
    "graveleau-waiting-profile",
    "asys-fujita-stationary",
    "allen-cahn",
    "asys-blowup-graveleau-pme-2d",
    "asys-hj-inverse",
)


@pytest.mark.parametrize("case", CASES)
def test_paper_config_and_local_source(case, tmp_path):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name=case, overrides=["run_id=config-check", f"run_root={tmp_path}"])
        OmegaConf.resolve(cfg)
    driver = OmegaConf.to_container(cfg.driver, resolve=True)
    for role in (None, "solver", "eval", "optimizer"):
        resolved = _driver_cfg_for_role(driver, role)
        assert resolved["model"] == "gpt-6.1-sol"
        assert resolved["reasoning_effort"] == "xhigh"
    assert not cfg.application.get("github_url")
    assert (ROOT / cfg.application.path).is_dir()
    problem = RepoTaskProblem.from_config(
        OmegaConf.to_container(cfg.application, resolve=True),
        cache_root=tmp_path / "cache",
        search_root=ROOT,
    )
    problem.copy_base_repo(tmp_path / "solver")
    task_roots = {
        "asys-fujita-ivp": "blowup/fujita_radial_n5_p3",
        "asys-pinn-allen-cahn-2d": "pinn/allen_cahn_2d",
        "asys-blowup-graveleau-pme-2d": "blowup/graveleau_pme_2d",
        "asys-hj-inverse": "inverse/hj_coefficients",
    }
    if case in task_roots:
        problems = tmp_path / "solver/problems"
        assert {str(path.relative_to(problems)) for path in problems.glob("*/*")} == {
            task_roots[case]
        }
    for path in cfg.application.editable.files:
        assert (tmp_path / "solver" / path).is_file()
    assert (ROOT / cfg.optimizer.initial_guidance).is_dir()
    assert _load_solver_worker_configs(cfg, search_root=ROOT)
    instantiate(cfg.optimizer.workers.selection, _convert_="all")
    evaluator = instantiate(cfg.optimizer.evaluation, _convert_="all")
    failure = OmegaConf.to_container(cfg.evaluation.failure_score)
    boundary = OmegaConf.to_container(cfg.application.boundary_failure_score)
    _validate_score_shape(boundary, failure)
    for name in ("solver_examples", "produced_optimizers"):
        sampler = instantiate(cfg.loop.sampling[name], _convert_="all")
        assert sampler.sample(["a", "b"], [failure, boundary], 1)
    evaluator.update({"a": {"elo": 1500.0}, "b": {"elo": 1500.0}}, {"a": failure, "b": boundary})
    for step in cfg.evaluation.steps:
        if isinstance(step, str):
            assert (ROOT / step).is_file()
            subprocess.run(["bash", "-n", str(ROOT / step)], check=True)
        else:
            assert (ROOT / step.prompt).is_dir()


def test_reference_fixtures_and_reuse(tmp_path):
    import sys

    script = ROOT / "scripts/prepare_references.sh"
    env = {**os.environ, "ASYS_PYTHON": sys.executable}
    subprocess.run(["bash", str(script), "--check"], env=env, check=True)
    target = tmp_path / "gclm"
    target.mkdir()
    (target / "field.npz").write_bytes(b"fixture")
    import hashlib

    digest = hashlib.sha256(b"fixture").hexdigest()
    (target / "SHA256SUMS").write_text(f"{digest}  ./field.npz\n")
    subprocess.run(["bash", str(script), "gclm", str(tmp_path)], env=env, check=True)
    (target / "field.npz").write_bytes(b"corrupt")
    assert subprocess.run(["bash", str(script), "gclm", str(tmp_path)], env=env).returncode != 0
    (target / "SHA256SUMS").rename(target / "SHA256SUMS.tmp")
    assert subprocess.run(["bash", str(script), "gclm", str(tmp_path)], env=env).returncode != 0
    old = tmp_path / "fujita"
    old.mkdir()
    payload = b'{"qualified": true, "refinement_evidence": {"solves": []}}'
    (old / "fujita-forward.json").write_bytes(payload)
    (old / "SHA256SUMS").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  ./fujita-forward.json\n"
    )
    assert subprocess.run(["bash", str(script), "fujita", str(tmp_path)], env=env).returncode != 0


def test_gclm_log_radius_mean_on_nonuniform_grid():
    import runpy

    import numpy as np

    assessor = runpy.run_path(
        str(CONFIGS / "application/asys-gclm-profile/kernel/common_single.py")
    )
    t = np.array([0.0, 0.2, 1.0, 3.0])
    assert assessor["_mean"](t, 2 * t + 1) == pytest.approx(4.0)


def test_reference_reuse_rejects_partial_checksum_list(tmp_path):
    import hashlib
    import json
    import sys

    target = tmp_path / "allen-cahn"
    target.mkdir()
    checksums = []
    for n, dt, count in ((384, 1 / 512, 262), (512, 1 / 1024, 12)):
        name = f"etdrk4-n{n}.json"
        payload = json.dumps(
            dict(
                method="ETDRK4",
                cells=n,
                maximum_dt=dt,
                time_knots=count,
                initial_regularizer=1e-12,
                contour_points=32,
            )
        ).encode()
        (target / name).write_bytes(payload)
        (target / f"etdrk4-n{n}.npz").write_bytes(b"corrupt")
        checksums.append(f"{hashlib.sha256(payload).hexdigest()}  ./{name}\n")
    (target / "SHA256SUMS").write_text("".join(checksums))
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/prepare_references.sh"), "allen-cahn", str(tmp_path)],
        env={**os.environ, "ASYS_PYTHON": sys.executable},
    )
    assert result.returncode != 0


@pytest.mark.parametrize("custom", [False, True])
def test_etdrk4_initial_condition_metadata(custom):
    import runpy

    import numpy as np

    reference = runpy.run_path(str(ROOT / "scripts/references/allen_cahn.py"))
    initial = (lambda x, y: np.full_like(x, 0.4)) if custom else None
    axis, values, metadata = reference["integrate_etdrk4"](
        8, np.array([0.0, 1 / 32]), 1 / 128, initial=initial
    )
    assert metadata["initial_regularizer"] == (None if custom else 1e-12)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    expected = initial(x, y) if custom else reference["initial_condition"](x, y, 1e-12)
    np.testing.assert_allclose(values[0], expected, rtol=0, atol=1e-15)
    if custom:
        np.testing.assert_allclose(values[-1], reference["reaction"](0.4, 1 / 32), atol=1e-10)
