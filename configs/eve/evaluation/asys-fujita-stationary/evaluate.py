"""Public equation/constraint fitting and scoring for the stationary Fujita profile."""
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

KEYS = ("physics", "ic", "ic_tangent", "bc")
FIT_SECONDS = 300
SOURCE_FILES = ("model.py", "loss.py", "optim.py", "config.yaml", "problem.py")


class FitDeadline(Exception):
    pass


def deadline(_signal, _frame):
    raise FitDeadline()


def source_hashes(solver):
    return {name: hashlib.sha256((solver / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def numerical_setup(seed):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    random.seed(seed)


def protected_problem(solver):
    # A separate namespace is never handed to candidate loss/optimizer code.
    spec = importlib.util.spec_from_file_location("_asys_bf_equations", solver / "problem.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def field_fingerprint(model, points):
    model.eval()
    points = points.detach().clone().requires_grad_(True)
    value = model(points)
    if value.shape != (len(points), 1):
        raise ValueError("Candidate must return a pointwise (N, 1) tensor")

    def derivative(y):
        result = (torch.autograd.grad(y.sum(), points, create_graph=True, allow_unused=True)[0]
                  if y.requires_grad else None)
        return points * 0 if result is None else result

    first = derivative(value)
    second = torch.stack([derivative(first[:, j:j+1]) for j in range(points.shape[1])], 1)
    values = {"value": value, "first": first, "second": second}
    if not all(bool(x.isfinite().all()) for x in values.values()):
        raise ValueError("nonfinite replay field or derivatives")
    return {name: tensor.detach().tolist() for name, tensor in values.items()}


def score_reconstruction(request_path, result_path):
    request = json.loads(request_path.read_text())
    solver, state = Path(request["solver"]), Path(request["state"])
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != request["evaluator_sha256"]:
        raise ValueError("protected evaluator changed before reconstruction")
    if source_hashes(solver) != request["source_sha256"]:
        raise ValueError("frozen source changed before reconstruction")
    if hashlib.sha256(state.read_bytes()).hexdigest() != request["state_sha256"]:
        raise ValueError("saved state changed before reconstruction")
    numerical_setup(request["fit_seed"])
    os.chdir(solver)
    sys.path.insert(0, str(solver))
    model = importlib.import_module("model").Candidate().double().cpu()
    saved = torch.load(state, map_location="cpu", weights_only=True)
    model.load_state_dict(saved, strict=True)
    # Reload immutable math after candidate import/construction/state restoration;
    # candidate's `import problem` is a different, mutable module object.
    problem = protected_problem(solver)
    torch.set_default_dtype(torch.float64)
    probe = field_fingerprint(model, problem.points(32, 171923)[0])
    score = public_scores(model, problem, problem.points(2048, 920614), admit=request["admit"])
    if source_hashes(solver) != request["source_sha256"]:
        raise ValueError("frozen source changed during reconstruction")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != request["evaluator_sha256"]:
        raise ValueError("protected evaluator changed during reconstruction")
    result = {"score": score, "probe": probe, "problem": problem.NAME,
              "state_sha256": request["state_sha256"], "state_keys": sorted(saved),
              "source_sha256": request["source_sha256"]}
    result_path.write_text(json.dumps(result, indent=2) + "\n")


def isolated_score(solver, state, output, label, seed, frozen_hashes, evaluator_hash, *, admit):
    request = {"solver": str(solver), "state": str(state), "fit_seed": seed,
               "source_sha256": frozen_hashes,
               "evaluator_sha256": evaluator_hash,
               "state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(), "admit": admit}
    request_path, result_path = output / f"{label}_replay_request.json", output / f"{label}_replay.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--replay-request", str(request_path), "--replay-output", str(result_path)],
                   check=True, timeout=90, cwd=solver,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return json.loads(result_path.read_text())


def public_scores(model, problem, monitor, *, admit=True):
    model.eval()
    if admit and hasattr(problem, "check_admissibility"):
        problem.check_admissibility(model, monitor)
    values = {key: float(value.detach()) for key, value in
              problem.components(model, *monitor).items()}
    if set(values) != set(KEYS) or not all(math.isfinite(v) and v >= 0 for v in values.values()):
        raise ValueError("nonfinite or malformed equation/constraint scores")
    return {key: -value for key, value in values.items()}


def evaluate(solver, output, *, require_admission=True, fit_seconds=FIT_SECONDS):
    solver, output = solver.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "score.yaml"
    admission = yaml.safe_load(score_path.read_text()) if score_path.exists() else {}
    if require_admission and admission.get("semantic_pass") is not True:
        raise ValueError("symbolic semantic admission did not pass")
    frozen_hashes = source_hashes(solver)
    evaluator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    seed = int(os.environ.get("ASYS_BF_FIT_SEED", "20261009"))
    numerical_setup(seed)
    os.chdir(solver)
    sys.path.insert(0, str(solver))
    problem = importlib.import_module("problem")
    model_module = importlib.import_module("model")
    config = yaml.safe_load((solver / "config.yaml").read_text())
    n = int(config.get("train_points_per_stratum", 256))
    max_steps = int(config.get("max_steps", 100000))
    if not 32 <= n <= 4096 or not 1 <= max_steps <= 1000000:
        raise ValueError("training points/steps outside declared CPU envelope")
    model = model_module.Candidate().double().cpu()
    loss_module = importlib.import_module("loss")
    optim_module = importlib.import_module("optim")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Candidate requires at least one trainable parameter")
    training = problem.points(n, seed)
    # Validate pointwise semantics and genuine parameter dependence on ordinary points.
    probe = training[0][:8]
    prediction = problem.field(model, probe)
    single = torch.cat([problem.field(model, p.reshape(1, -1)) for p in probe])
    if not torch.allclose(prediction, single, atol=1e-10, rtol=1e-9):
        raise ValueError("Candidate output must be independent of batch composition")
    dependent = False
    for value in prediction[:, 0]:
        dependency = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        dependent |= any(g is not None and bool(torch.isfinite(g).all()) and bool(g.abs().sum() > 0)
                         for g in dependency)
    if not dependent:
        raise ValueError("Candidate output does not depend on trainable parameters at the probe")
    torch.save(model.state_dict(), output / "initial_state_dict.pt")
    initial = isolated_score(solver, output / "initial_state_dict.pt", output, "initial",
                             seed, frozen_hashes, evaluator_hash, admit=False)
    helpers = {**problem.helpers(), "config": config}
    stepper = optim_module.build_stepper(params, helpers)
    if not callable(getattr(stepper, "step", None)) or not callable(getattr(stepper, "zero_grad", None)):
        raise ValueError("build_stepper must provide step(closure) and zero_grad()")
    steps = 0
    best_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    losses = []
    started = time.monotonic()
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, fit_seconds)
    reason = "max_steps"
    try:
        for iteration in range(max_steps):
            model.train()

            def closure():
                nonlocal best_loss, best_state
                stepper.zero_grad()
                loss = loss_module.loss_fn(model, *training, helpers)
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise ValueError("loss_fn must return a finite differentiable scalar tensor")
                loss.backward()
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value
                    best_state = copy.deepcopy(model.state_dict())
                return loss

            stepper.step(closure)
            steps = iteration + 1
            if steps == 1 or steps % 100 == 0:
                losses.append({"step": steps, "best_training_loss": best_loss,
                               "seconds": time.monotonic() - started})
    except FitDeadline:
        reason = "fit_time_ceiling"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    fit_elapsed = time.monotonic() - started
    if not math.isfinite(best_loss):
        raise ValueError("no completed finite differentiable fitting closure")
    model.load_state_dict(best_state)
    torch.save(best_state, output / "state_dict.pt")
    if source_hashes(solver) != frozen_hashes:
        raise ValueError("frozen source changed during fitting")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != evaluator_hash:
        raise ValueError("protected evaluator changed during fitting")
    clean = protected_problem(solver)
    fit_probe = field_fingerprint(model, clean.points(32, 171923)[0])
    receipt = {
        "problem": clean.NAME, "fit_seed": seed, "monitor_seed": 920614,
        "fit_ceiling_seconds": fit_seconds, "fit_seconds": fit_elapsed,
        "completed_steps": steps, "stopping_reason": reason,
        "training_points_per_stratum": n, "monitor_points_per_stratum": 2048,
        "training_loss": best_loss, "source_sha256": frozen_hashes,
        "evaluator_sha256": evaluator_hash,
        "state_sha256": hashlib.sha256((output / "state_dict.pt").read_bytes()).hexdigest(),
        "initial_state_sha256": hashlib.sha256((output / "initial_state_dict.pt").read_bytes()).hexdigest(),
        "python": sys.executable, "torch_version": torch.__version__,
        "trainable_scalars": sum(p.numel() for p in params),
        "buffer_scalars": sum(b.numel() for b in model.buffers()),
        "state_keys": sorted(best_state), "state_bytes": (output / "state_dict.pt").stat().st_size,
        "scoring_protocol": "isolated_frozen_source_state_v2",
    }
    (output / "config.yaml").write_text((solver / "config.yaml").read_text())
    (output / "fit_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "training_trace.json").write_text(json.dumps(losses, indent=2) + "\n")
    (output / "initial_public_scores.json").write_text(json.dumps(initial["score"], indent=2) + "\n")
    (output / "fitting_field_probe.json").write_text(json.dumps(fit_probe, indent=2) + "\n")
    replay = isolated_score(solver, output / "state_dict.pt", output, "final", seed,
                            frozen_hashes, evaluator_hash, admit=True)
    matches = all(torch.allclose(torch.tensor(fit_probe[key]), torch.tensor(replay["probe"][key]),
                                 atol=1e-10, rtol=1e-9) for key in fit_probe)
    (output / "replay_comparison.json").write_text(json.dumps({"passed": matches}, indent=2) + "\n")
    if not matches:
        raise ValueError("fitted field/derivatives are not reconstructible from saved model state")
    final = replay["score"]
    final.update(semantic_pass=admission.get("semantic_pass") is True,
                 representation_kind=admission.get("representation_kind", "not_reviewed"),
                 summary=f"{clean.NAME}; isolated source/state equation/constraint MSE; {reason}")
    score_path.write_text(yaml.safe_dump(final, sort_keys=False))
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-request", type=Path)
    parser.add_argument("--replay-output", type=Path)
    args = parser.parse_args()
    if args.replay_request:
        score_reconstruction(args.replay_request, args.replay_output)
        raise SystemExit(0)
    destination = Path(os.environ["EVE_EVAL_LOG_ROOT"])
    destination.mkdir(parents=True, exist_ok=True)
    try:
        print(json.dumps(evaluate(Path(os.environ["EVE_SOLVER_ROOT"]), destination)))
    except Exception as exc:
        result = {key: -1e9 for key in KEYS}
        result.update(semantic_pass=False, representation_kind="invalid",
                      summary=f"{type(exc).__name__}: {exc}")
        (destination / "error.txt").write_text(result["summary"] + "\n")
        (destination / "score.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
        print(json.dumps(result))
