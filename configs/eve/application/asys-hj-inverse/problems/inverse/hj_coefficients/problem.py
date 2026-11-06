"""Hamilton–Jacobi field fitting and public validation."""

import copy
import hashlib
import importlib.util
import json
import math
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from asys_task.evaluate import validate_sources


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initial(points, problem):
    modes = points.new_tensor(problem["initial_modes"])
    weights = modes[:, 2] / (modes[:, :2].abs().sum(1) + 1) / problem["initial_scale"]
    return (torch.sin(points[:, :2] @ modes[:, :2].T + modes[:, 3]) @ weights)[:, None]


def derivative(values, points):
    if not values.requires_grad:
        return points * 0
    grad = torch.autograd.grad(values.sum(), points, create_graph=True, allow_unused=True)[0]
    return points * 0 if grad is None else grad


class Field(nn.Module):
    def __init__(self, candidate, problem):
        super().__init__()
        self.candidate = candidate.double()
        self.mode = problem["mode"]
        bounds = torch.tensor(problem["coefficient_bounds"], dtype=torch.float64)
        self.register_buffer("lower", bounds[:, 0])
        self.register_buffer("span", bounds[:, 1] - bounds[:, 0])
        if self.mode == "inverse":
            start = torch.tensor(problem["coefficient_start"], dtype=torch.float64)
            self.raw = nn.Parameter(torch.logit((start - self.lower) / self.span))
        else:
            self.register_buffer("known", torch.tensor(problem["known_coefficients"], dtype=torch.float64))

    def coefficients(self):
        return self.known if self.mode == "forward" else self.lower + self.span * self.raw.sigmoid()

    def forward(self, points):
        linked = getattr(self.candidate, "forward_with_coefficients", None)
        value = self.candidate(points) if linked is None else linked(points, self.coefficients())
        if value.shape != (len(points), 1):
            raise ValueError("Candidate must return [N,1]")
        return value

    def residual(self, points):
        p = points.detach().clone().requires_grad_(True)
        grad = derivative(self(p), p)
        uxx = derivative(grad[:, :1], p)[:, :1]
        uyy = derivative(grad[:, 1:2], p)[:, 1:2]
        q = self.coefficients()
        return grad[:, 2:3] - q[0] * uxx - q[1] * uyy - q[2] * grad[:, :1] ** 2 - q[3] * grad[:, 1:2] ** 2


def observations(path, horizon):
    with np.load(path, allow_pickle=False) as data:
        expected = {f"{split}_{kind}" for split in ("train", "validation") for kind in ("points", "values")}
        if set(data.files) != expected:
            raise ValueError("Observation-only NPZ schema mismatch")
        arrays = {key: np.array(data[key], copy=True) for key in expected}
    for split in ("train", "validation"):
        p, v = arrays[f"{split}_points"], arrays[f"{split}_values"]
        if p.shape != (len(p), 3) or v.shape != (len(p), 1) or not len(p):
            raise ValueError("Invalid observation shapes")
        if not np.isfinite(p).all() or not np.isfinite(v).all():
            raise ValueError("Nonfinite observations")
        if np.any(p < 0) or np.any(p[:, :2] >= 2 * np.pi) or np.any(p[:, 2] > horizon + 1e-12):
            raise ValueError("Observation outside declared domain")
    if arrays["train_points"][:, 2].max() >= arrays["validation_points"][:, 2].min():
        raise ValueError("Validation must follow training times")
    return {key: torch.tensor(value, dtype=torch.float64) for key, value in arrays.items()}


def batch(problem, fit, generator, data=None, validation=False):
    p = torch.rand((fit["n_physics"], 3), generator=generator, dtype=torch.float64)
    p[:, :2] *= 2 * math.pi
    p[:, 2] *= problem["horizon"]
    ic = torch.rand((fit["n_ic"], 3), generator=generator, dtype=torch.float64)
    ic[:, :2] *= 2 * math.pi
    ic[:, 2] = 0
    bc = torch.rand((fit["n_bc"], 2), generator=generator, dtype=torch.float64)
    bc[:, 0] *= 2 * math.pi
    bc[:, 1] *= problem["horizon"]
    result = {"physics": p, "initial": ic, "boundary": bc}
    if data is not None:
        split = "validation" if validation else "train"
        points, values = data[f"{split}_points"], data[f"{split}_values"]
        if not validation:
            idx = torch.randint(len(points), (fit["n_observation"],), generator=generator)
            points, values = points[idx], values[idx]
        result["observations"] = (points, values)
    return result


def components(field, samples, problem):
    scale, horizon = problem["field_scale"], problem["horizon"]
    result = {"physics": (horizon * field.residual(samples["physics"]) / scale).square().mean()}
    p = samples["initial"]
    result["ic"] = ((field(p) - initial(p, problem)) / scale).square().mean()
    b = samples["boundary"]
    boundary = b.new_zeros(())
    for axis in (0, 1):
        left = torch.stack([b[:, 0] * 0, b[:, 0], b[:, 1]], dim=1)
        if axis:
            left = left[:, [1, 0, 2]]
        right = left.clone()
        right[:, axis] = 2 * math.pi
        left.requires_grad_(True)
        right.requires_grad_(True)
        ul, ur = field(left), field(right)
        dl, dr = derivative(ul, left)[:, axis], derivative(ur, right)[:, axis]
        boundary = boundary + ((ul - ur) / scale).square().mean() + (2 * math.pi * (dl - dr) / scale).square().mean()
    result["bc"] = boundary / 4
    result["observation"] = b.new_zeros(())
    if "observations" in samples:
        p, v = samples["observations"]
        result["observation"] = ((field(p) - v) / scale).square().mean()
    return result


def probe(field, horizon):
    banned = (nn.Linear, nn.modules.conv._ConvNd, nn.RNNBase, nn.MultiheadAttention, nn.Transformer)
    if any(isinstance(layer, banned) for layer in field.candidate.modules()):
        raise ValueError("Neural solution layers are prohibited")
    g = torch.Generator().manual_seed(403)
    p = torch.rand((8, 3), generator=g, dtype=torch.float64)
    p[:, :2] = 0.2 + 5.8 * p[:, :2]
    p[:, 2] *= 0.8 * horizon
    p.requires_grad_(True)
    u, grad = field(p), None
    if not torch.isfinite(u).all() or not torch.allclose(u, torch.cat([field(x[None]) for x in p]), rtol=1e-9, atol=1e-10):
        raise ValueError("Nonfinite or batch-dependent field")
    grad = derivative(u, p)
    for axis in range(3):
        step = 1e-5 if axis == 2 else 1e-4
        delta = p.new_zeros(p.shape)
        delta[:, axis] = step
        plus, minus = field(p + delta), field(p - delta)
        if not torch.allclose((plus - minus) / (2 * step), grad[:, axis:axis + 1], rtol=1e-4, atol=1e-5):
            raise ValueError("First-derivative mismatch")
        if axis < 2:
            second = derivative(grad[:, axis:axis + 1], p)[:, axis:axis + 1]
            if not torch.allclose((plus - 2 * u + minus) / step**2, second, rtol=1e-3, atol=1e-4):
                raise ValueError("Second-derivative mismatch")
    if not torch.isfinite(field.residual(p)).all():
        raise ValueError("Nonfinite physics")


def fit_field(solver, problem, contract, data, output):
    validate_sources(solver)
    config = yaml.safe_load((solver / "config.yaml").read_text())
    allowed = {"steps", "learning_rate", "n_physics", "n_ic", "n_bc", "n_observation"}
    if set(config) != allowed:
        raise ValueError("Unexpected candidate fitting settings")
    for key in allowed - {"learning_rate"}:
        if not isinstance(config[key], int) or not 1 <= config[key] <= contract[key]:
            raise ValueError(f"Candidate {key} exceeds declared budget")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("Invalid learning rate")
    torch.set_num_threads(contract["threads"])
    torch.manual_seed(contract["seed"])
    public = copy.deepcopy(problem)
    candidate = load_module(solver / "model.py", "hj_candidate").Candidate(public)
    field = Field(candidate, problem).eval()
    probe(field, problem["horizon"])
    parameters = [p for p in field.parameters() if p.requires_grad]
    steps = config["steps"] if parameters else 0
    if parameters:
        loss = load_module(solver / "loss.py", "hj_loss").training_loss
        optimizer = load_module(solver / "optim.py", "hj_optim").make_optimizer(parameters, config)
    generator = torch.Generator().manual_seed(contract["seed"] + 1)
    vg = torch.Generator().manual_seed(contract["seed"] + 100001)
    validation = [batch(problem, contract, vg, data, True) for _ in range(contract["validation_batches"])]
    best, state, selected, checkpoints = math.inf, None, None, []
    gradient_evaluations = 0
    started = time.monotonic()

    def checkpoint(step):
        nonlocal best, state, selected
        field.eval()
        rows = [{k: float(v.detach()) for k, v in components(field, b, problem).items()} for b in validation]
        scores = {k: sum(row[k] for row in rows) / len(rows) for k in rows[0]}
        total = sum(scores.values())
        if not math.isfinite(total):
            raise ValueError("Nonfinite public score")
        checkpoints.append({"step": step, "components": scores, "total": total, "seconds": time.monotonic() - started})
        if total < best:
            best, state, selected = total, copy.deepcopy(field.state_dict()), step
        write(output / "checkpoints.json", checkpoints)

    checkpoint(0)
    for step in range(1, steps + 1):
        if time.monotonic() - started > contract["wall_seconds"]:
            raise TimeoutError("Declared fit wall guard exhausted")
        field.train()
        samples = batch(problem, config, generator, data)

        def closure():
            nonlocal gradient_evaluations
            if time.monotonic() - started > contract["wall_seconds"]:
                raise TimeoutError("Declared fit wall guard exhausted")
            optimizer.zero_grad(set_to_none=True)
            objective = loss(components(field, samples, problem), config)
            if objective.ndim != 0 or not torch.isfinite(objective):
                raise ValueError("Invalid scalar training loss")
            objective.backward()
            gradient_evaluations += 1
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in field.parameters()):
                raise ValueError("Nonfinite gradients")
            return objective

        optimizer.step(closure)
        if step % contract["checkpoint_every"] == 0 or step == steps:
            checkpoint(step)
    field.load_state_dict(state)
    probe(field, problem["horizon"])
    torch.save(state, output / "selected-state.pt")
    restored = Field(load_module(solver / "model.py", "hj_replay").Candidate(public), problem).eval()
    restored.load_state_dict(torch.load(output / "selected-state.pt", map_location="cpu", weights_only=True))
    points = validation[0]["physics"]
    if not torch.equal(field(points), restored(points)):
        raise ValueError("Saved-state replay differs")
    return {"score": -best, "completed": True, "kind": "field", "selected_step": selected,
            "steps": steps, "gradient_evaluations": gradient_evaluations,
            "seconds": time.monotonic() - started,
            "coefficients": field.coefficients().detach().tolist(),
            "field_parameters": sum(p.numel() for p in candidate.parameters() if p.requires_grad),
            "fixed_parameter_scalars": sum(p.numel() for p in candidate.parameters() if not p.requires_grad),
            "buffer_scalars": sum(b.numel() for b in candidate.buffers()),
            "physical_parameters": 4 if problem["mode"] == "inverse" else 0}


def main():
    output = Path(os.environ["EVE_EVAL_LOG_ROOT"])
    solver = Path(os.environ["EVE_SOLVER_ROOT"]) / "problems/inverse/hj_coefficients"
    output.mkdir(parents=True, exist_ok=True)
    write(output / "score.yaml", {"score": -1e12, "summary": "HJ evaluation incomplete"})
    shutil.copy2(__file__, output / "evaluator.py")
    problem = json.loads((Path(__file__).parent / "problem.json").read_text())
    contract = json.loads((Path(__file__).parent / "fit.json").read_text())
    if problem["mode"] not in ("forward", "inverse") or ("known_coefficients" in problem) != (problem["mode"] == "forward"):
        raise ValueError("Invalid public HJ mode")
    data = observations(Path(__file__).parent / "observations.npz", problem["horizon"]) if problem["mode"] == "inverse" else None
    receipt = hashlib.sha256((Path(__file__).parent / "observations.npz").read_bytes()).hexdigest() if data is not None else None
    write(output / "context.json", {"problem": problem, "contract": contract, "kind": "field",
                                    "observations_sha256": receipt})
    submission = output / "submission"
    submission.mkdir(exist_ok=True)
    for name in ("model.py", "loss.py", "optim.py", "config.yaml"):
        shutil.copy2(solver / name, submission / name)
    result = fit_field(solver, problem, contract, data, output)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result["process_user_cpu_seconds"] = usage.ru_utime
    result["process_system_cpu_seconds"] = usage.ru_stime
    result["peak_rss_kib"] = usage.ru_maxrss
    write(output / "result.json", result)
    write(output / "score.yaml", {"score": result["score"], "summary": "HJ field completed", "completed": True})
    print(json.dumps(result))


if __name__ == "__main__":
    main()
