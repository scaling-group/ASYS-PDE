"""Scientific task scoring and workspace-check contracts, without agent calls."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import torch
import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs/eve"
CASES = (
    ("asys-pinn-allen-cahn-2d", "pinn/allen_cahn_2d"),
    ("asys-fujita-ivp", "blowup/fujita_radial_n5_p3"),
    ("asys-blowup-graveleau-pme-2d", "blowup/graveleau_pme_2d"),
)
SCORES = ("score", "physics", "ic", "ic_tangent", "bc")


def run_python(args, *, cwd, env=None):
    result = subprocess.run(
        [sys.executable, "-B", *map(str, args)],
        cwd=cwd,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("app,problem", CASES)
@pytest.mark.parametrize("mutator", ["loss", "optim"])
def test_fitting_cannot_replace_fixed_scores(app, problem, mutator, tmp_path):
    solver = tmp_path / "solver"
    shutil.copytree(CONFIGS / "application" / app, solver)
    case = solver / "problems" / problem
    config = yaml.safe_load((case / "config.yaml").read_text())
    config.update(max_steps=1)
    (case / "config.yaml").write_text(yaml.safe_dump(config))
    sources = {
        "loss": "def loss_fn(model, collocation, ic_points, bc_points, helpers):\n"
        "    return sum(p.square().sum() for p in model.parameters())\n",
        "optim": "import torch\ndef build_stepper(params, helpers):\n"
        "    return torch.optim.SGD(params, lr=0.0)\n",
    }
    for name, source in sources.items():
        (case / f"{name}.py").write_text(source)
    env = dict(
        ASYS_WALL_TIME="30",
        ASYS_NF="8",
        ASYS_NIC="8",
        ASYS_NBC="8",
        ASYS_DEVICE="cpu",
        ASYS_NX="16",
        PYTHONDONTWRITEBYTECODE="1",
    )
    state = tmp_path / "state.pt"
    command = [case / "evaluate.py", "--json", "--save-state-dict", state]
    expected = run_python(command, cwd=solver, env=env)
    # A fitting-time helper edit must not alter official scoring or diagnostics.
    mutation = (
        "    helpers.__class__.score = lambda self, model: "
        "dict(score=0., physics=0., ic=0., ic_tangent=0., bc=0.)\n"
        "    helpers.__class__.diagnostics = lambda self, model: dict(fake=True)\n"
    )
    (case / f"{mutator}.py").write_text(
        sources[mutator].replace("    return ", mutation + "    return ")
    )
    actual = run_python(command, cwd=solver, env=env)
    for key in SCORES:
        assert actual[key] == pytest.approx(expected[key], rel=1e-12, abs=1e-12)
    assert any(actual[key] < -1e-12 for key in SCORES)
    assert actual["diagnostics"] == expected["diagnostics"]
    assert actual["score_components"]["train_n_collocation"] == 8
    assert actual["score_components"]["score_n_collocation"] > 8
    assert state.is_file()

    for name in sources:
        (case / f"{name}.py").write_text("raise AssertionError('Fitting source imported')\n")
    replay = run_python([case / "evaluate.py", "--score-state", state], cwd=solver, env=env)
    for key in SCORES:
        assert replay[key] == actual[key]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("app,problem", CASES)
def test_scalar_tasks_reject_multicolumn_output(app, problem, tmp_path):
    solver = tmp_path / "solver"
    shutil.copytree(CONFIGS / "application" / app, solver)
    case = solver / "problems" / problem
    (case / "model.py").write_text(
        "import torch\nclass Candidate(torch.nn.Module):\n"
        "    def __init__(self, config=None):\n        super().__init__()\n"
        "        self.scale = torch.nn.Parameter(torch.ones(()))\n"
        "    def forward(self, points):\n        return self.scale * points[:, :2]\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", str(case / "evaluate.py"), "--json"],
        cwd=solver,
        env={**os.environ, "ASYS_WALL_TIME": "0", "ASYS_DEVICE": "cpu"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "one finite scalar column" in result.stderr


JET_SOURCE = """import torch
from problem import initial, initial_rhs

class Candidate(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.log_width = torch.nn.Parameter(torch.tensor(-14.0))

    def forward(self, points):
        if not points.requires_grad:
            points = points.detach().clone().requires_grad_(True)
        t = points[:, 2:3]
        jet = initial(points) + t * initial_rhs(points)
        return torch.exp(-(t / self.log_width.exp()).square()) * jet
"""


@pytest.fixture
def geometric_ac():
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    root = CONFIGS / "application/allen-cahn"
    problem = load_module("_test_ac_problem", root / "problem.py")
    evaluator = load_module("_test_ac_evaluate", root / "evaluate.py")
    yield problem, evaluator
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


def test_initial_secant_distinguishes_collapsed_and_retained_jets(geometric_ac):
    problem, evaluator = geometric_ac

    class Jet(torch.nn.Module):
        def __init__(self, collapsed):
            super().__init__()
            self.collapsed = collapsed
            self.log_width = torch.nn.Parameter(torch.tensor(-14.0))

        def forward(self, points):
            if not points.requires_grad:
                points = points.detach().clone().requires_grad_(True)
            t = points[:, 2:3]
            jet = problem.initial(points) + t * problem.initial_rhs(points)
            return jet * torch.exp(-(t / self.log_width.exp()).square()) if self.collapsed else jet

    collapsed, retained = Jet(True), Jet(False)
    p, i, b = problem.sample((16, 16, 8), torch.Generator().manual_seed(701))
    # The training helper still measures the exact AD jet; it is not an admission gate.
    assert float(problem.terms(collapsed, p, i, b)["ic_tangent"].detach()) < 1e-20
    collapsed_score = evaluator.public_score(problem, collapsed)
    retained_score = evaluator.public_score(problem, retained)
    assert collapsed_score["ic"] == pytest.approx(0.0, abs=1e-20)
    assert collapsed_score["physics"] == pytest.approx(0.0, abs=1e-20)
    assert collapsed_score["ic_tangent"] < -1000
    assert retained_score["ic_tangent"] == pytest.approx(0.0, abs=1e-20)


def test_geometric_ac_checkpoint_replay_uses_initial_secant(tmp_path):
    solver = tmp_path / "solver"
    shutil.copytree(CONFIGS / "application/allen-cahn", solver)
    (solver / "model.py").write_text(JET_SOURCE)
    config = yaml.safe_load((solver / "config.yaml").read_text())
    config["max_steps"] = 0
    (solver / "config.yaml").write_text(yaml.safe_dump(config))
    env = dict(ASYS_NUM_THREADS="1", ASYS_WALL_TIME="30")
    checkpoint = solver / "state/checkpoint.pt"
    command = [solver / "evaluate.py", "--workspace", solver]
    fitted = run_python(
        [*command, "--log", tmp_path / "fit", "--checkpoint-out", checkpoint],
        cwd=solver,
        env=env,
    )
    replay = run_python([*command, "--log", tmp_path / "replay"], cwd=solver, env=env)
    assert -1e9 < fitted["ic_tangent"] < -1000
    assert fitted == replay
    manifest = json.loads((tmp_path / "replay/manifest.json").read_text())
    assert manifest["operation"] == "checkpoint_replay"
    assert manifest["optimizer_steps_completed"] == manifest["scoring_optimizer_steps"] == 0


@pytest.mark.parametrize("app,problem", [CASES[0], CASES[2]])
@pytest.mark.parametrize("failure", [None, "model", "loss", "optim", "import"])
def test_workspace_checker_never_calls_official_evaluator(app, problem, failure, tmp_path):
    solver = tmp_path / "solver"
    shutil.copytree(CONFIGS / "application" / app, solver)
    for path in (solver / "problems" / problem / "evaluate.py", solver / "asys_task/evaluate.py"):
        path.write_text("raise AssertionError('Official evaluator invoked by local checker')\n")
    if failure == "model":
        (solver / "problems" / problem / "model.py").write_text(
            "import torch\nclass Candidate(torch.nn.Module):\n"
            "    def __init__(self, config=None):\n        super().__init__()\n"
            "    def forward(self, points):\n        return points[:, :1] * float('nan')\n"
        )
    elif failure == "loss":
        (solver / "problems" / problem / "loss.py").write_text(
            "def loss_fn(model, collocation, ic_points, bc_points, helpers):\n"
            "    return model(collocation).mean().detach()\n"
        )
    elif failure == "optim":
        (solver / "problems" / problem / "optim.py").write_text(
            "def build_stepper(params, helpers):\n    raise ValueError('Broken optimizer')\n"
        )
    elif failure == "import":
        path = solver / "problems" / problem / "loss.py"
        path.write_text("from functools import reduce\n" + path.read_text())
    path = CONFIGS / "optimizer" / app / "immutable/.codex/agents/check-runner.toml"
    instructions = tomllib.loads(path.read_text())["developer_instructions"]
    shell = instructions.split('export EVE_WORKSPACE_ROOT="$(pwd)"', 1)[1]
    shell = 'export EVE_WORKSPACE_ROOT="$(pwd)"' + shell.split("\nRead `${LOG_PATH}`", 1)[0]
    # The real boundary checker is tested elsewhere; exercise this task's local checks.
    shell = shell.replace("{{BOUNDARY_CHECK_COMMAND}}", "true") + '\nexit "$rc"\n'
    result = subprocess.run(
        ["bash"],
        input=shell,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "ASYS_PYTHON": sys.executable, "ASYS_PROBLEM": problem},
    )
    log = next((tmp_path / "logs/optimize").glob("asys_blowup_check_*.log")).read_text()
    assert (result.returncode == 0) == (failure is None), result.stderr + log
    assert ("interface_ok" in log) == (failure is None)
    assert "Official evaluator invoked" not in log
