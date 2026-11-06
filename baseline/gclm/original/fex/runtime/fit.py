"""Reference-free fitting and independent public evaluation of a gCLM profile."""

import argparse
import ast
import copy
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from operators import NormalizedProfile, Operators, losses, terms


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_source(application):
    """Basic execution boundary."""
    forbidden = {"open", "eval", "exec", "compile", "__import__", "getattr", "setattr"}
    blocked_methods = {"detach", "numpy", "load", "save", "set_grad_enabled", "no_grad"}
    for name in ("model.py", "loss.py", "optim.py"):
        tree = ast.parse((application / name).read_text(), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name.split(".")[0] not in {"torch", "math"} for alias in node.names
            ):
                raise ValueError(f"Unsupported import in {name}")
            if isinstance(node, ast.ImportFrom) and (
                not node.module or node.module.split(".")[0] not in {"torch", "math", "__future__"}
            ):
                raise ValueError(f"Unsupported import in {name}")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in forbidden:
                    raise ValueError(f"Disallowed execution or file access in {name}")
                if isinstance(node.func, ast.Attribute) and node.func.attr in blocked_methods:
                    raise ValueError(f"Disallowed gradient/state operation in {name}")


class Helpers:
    def __init__(self, nodes):
        self.operators = Operators(nodes)

    def normalize(self, model):
        return NormalizedProfile(model, self.operators)

    def losses(self, profile, y):
        return losses(profile, y, self.operators)


def check_profile(model, nodes=256):
    profile = NormalizedProfile(model, Operators(nodes))
    y = torch.logspace(-3, 3, 193)
    beta = float(model.beta())
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("A finite positive collapsing exponent is required")
    f = profile(y)
    if f.shape != y.shape or not bool(torch.isfinite(f).all() & (f > 0).all()):
        raise ValueError("Profile must be finite, positive on y>0 and shape-preserving")
    odd_error = float(torch.max(abs(profile(-y) + f)) / torch.max(abs(f)))
    singleton = torch.stack([profile(v.reshape(1))[0] for v in y[::16]])
    batch_error = float(torch.max(abs(singleton - f[::16])) / torch.max(abs(f)))
    repeated_error = float(torch.max(abs(profile(y) - f)) / torch.max(abs(f)))
    zero = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    fzero = profile(zero)
    slope = torch.autograd.grad(fzero.sum(), zero)[0]
    origin_error = max(abs(float(fzero)), abs(float(slope) - 1))
    probes = torch.tensor([0.003, 0.03, 0.3, 1.0, 3.0, 30.0, 300.0], requires_grad=True)
    exact_derivative = torch.autograd.grad(profile(probes).sum(), probes)[0]
    step = 1e-5 * probes.detach().clamp_min(1)
    fd = (profile(probes + step) - profile(probes - step)) / (2 * step)
    derivative_error = float(torch.max(abs(fd - exact_derivative) / (1 + abs(exact_derivative))))
    if max(odd_error, batch_error, repeated_error, origin_error) > 1e-9:
        raise ValueError("Oddness, gauge or pointwise/repeatability check failed")
    if derivative_error > 1e-6:
        raise ValueError("Coordinate derivative check failed")
    return profile, dict(
        beta=beta,
        normalization_amplitude=float(profile.amplitude),
        normalization_coordinate_scale=float(profile.scale),
        odd_error=odd_error,
        batch_error=batch_error,
        repeated_error=repeated_error,
        origin_error=origin_error,
        derivative_error=derivative_error,
    )


def public_score(model):
    profile, checks = check_profile(model)
    y = torch.logspace(-3, 3, 193)
    measured = []
    for nodes in (256, 512):
        op = Operators(nodes)
        pieces = terms(profile, model.beta(), y, op, create_graph=False)
        residual = pieces.sum(dim=0)
        pointwise_squared = residual.square() / pieces.square().sum(dim=0).clamp_min(1e-28)
        center_error = abs(float(op.center(profile)) - 1)
        measured.append(
            dict(
                nodes=nodes,
                pointwise_rms=float(torch.sqrt(pointwise_squared.mean())),
                component_rms=float(torch.sqrt(residual.square().sum() / pieces.square().sum())),
                maximum_relative=float(torch.max(abs(residual) / pieces.abs().sum(dim=0))),
                center_error=center_error,
            )
        )
    fine = measured[-1]
    change = abs(fine["pointwise_rms"] - measured[0]["pointwise_rms"])
    resolved = change <= max(1e-8, 0.005 * fine["pointwise_rms"])
    resolved = resolved and fine["center_error"] <= 1e-6
    # At the far end, a decaying similarity profile must have logarithmic slope
    # tending to -1/beta. Keep the diagnostic separate from finite-box residuals.
    tail_y = torch.logspace(3, 5, 9, requires_grad=True)
    tail = profile(tail_y)
    tail_derivative = torch.autograd.grad(tail.sum(), tail_y)[0]
    tail_slope = tail_y * tail_derivative / tail
    tail_error = float(torch.max(abs(tail_slope + 1 / model.beta())))
    admissible = resolved and bool(torch.isfinite(tail_slope).all() & (tail > 0).all())
    admissible = admissible and tail_error <= 0.05
    return dict(
        admissible=admissible,
        score=-max(fine["pointwise_rms"], 1e-8) if admissible else -1e9,
        profile_checks=checks,
        refinements=measured,
        refinement_change=change,
        tail_log_slope_error=tail_error,
        parameters=sum(p.numel() for p in model.parameters()),
        buffers=sum(p.numel() for p in model.buffers()),
        normalized_profile="f(y)=raw_model(coordinate_scale*y)/amplitude; both constants frozen",
    )


def fit(application, output, wall_seconds=300, max_updates_override=None, checkpoints=()):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    check_source(application)
    config = yaml.safe_load((application / "config.yaml").read_text())
    model = load(application / "model.py", "profile_model").Model(config.get("model", {})).double()
    objective = load(application / "loss.py", "profile_loss").loss_fn
    build_optimizer = load(application / "optim.py", "profile_optim").build_optimizer
    training = config.get("training", {})
    count = int(training.get("collocation", 96))
    nodes = int(training.get("quadrature", 128))
    updates = int(training.get("max_updates", 100000))
    if not (32 <= count <= 256 and 64 <= nodes <= 512 and 1 <= updates <= 100000):
        raise ValueError("Training count/quadrature/update bounds exceeded")
    if max_updates_override is not None:
        updates = min(updates, max_updates_override)
    helpers = Helpers(nodes)
    # Offset interior training sites from the fixed public grid; no references.
    q = (torch.arange(count, dtype=torch.float64) + 0.5) / count
    batch = torch.exp((2 * q - 1) * math.log(1000))
    if not any(p.requires_grad for p in model.parameters()):
        score = public_score(model)
        torch.save(model.state_dict(), output / "state_dict.pt")
        (output / "public.json").write_text(json.dumps(score, indent=2, allow_nan=False) + "\n")
        result = dict(
            completed=True,
            best_training_loss=None,
            loss_evaluations=0,
            completed_updates=0,
            stop="no_trainable_parameters",
            fit_seconds=0.0,
            scoring_seconds=0.0,
            total_seconds=time.monotonic() - started,
            checkpoints=[],
            initial_seed=int(config.get("model", {}).get("seed", 0)),
            training_config=config,
            public=score,
            trace=[],
        )
        (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    if type(optimizer) not in {
        torch.optim.LBFGS,
        torch.optim.Adam,
        torch.optim.AdamW,
        torch.optim.SGD,
    }:
        raise ValueError("Optimizer must be an ordinary allowed Torch optimizer")
    warmup_fraction = float(training.get("warmup_fraction", 0))
    if not 0 <= warmup_fraction <= 0.5:
        raise ValueError("Warmup fraction must lie in [0,0.5]")
    warmup = torch.optim.Adam(model.parameters(), lr=float(training.get("warmup_lr", 0.001)))
    best_loss = math.inf
    best_state = None
    evaluations = 0
    trace = []

    def closure():
        nonlocal best_loss, best_state, evaluations
        model.zero_grad(set_to_none=True)
        value = objective(model, helpers, batch, config.get("loss", {}))
        if value.ndim or not bool(torch.isfinite(value)):
            raise ValueError("Training loss must be a finite scalar")
        value.backward()
        if any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        ):
            raise ValueError("Nonfinite parameter gradient")
        number = float(value)
        evaluations += 1
        if number < best_loss:
            best_loss = number
            best_state = copy.deepcopy(model.state_dict())
        return value

    fit_started = time.monotonic()
    excluded_scoring_seconds = 0.0

    def elapsed():
        return time.monotonic() - fit_started - excluded_scoring_seconds

    saved = []
    pending = list(sorted(set(float(v) for v in checkpoints)))
    closure()
    stop = "max_updates"
    for update in range(1, updates + 1):
        if elapsed() >= wall_seconds:
            stop = "wall_time"
            break
        if elapsed() < warmup_fraction * wall_seconds:
            closure()
            warmup.step()
        else:
            optimizer.step(closure)
        if update % 10 == 0:
            trace.append(dict(update=update, fit_seconds=elapsed(), best_training_loss=best_loss))
        if pending and elapsed() >= pending[0]:
            scoring_start = time.monotonic()
            checkpoint_fit_seconds = elapsed()
            current = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            while pending and checkpoint_fit_seconds >= pending[0]:
                budget = pending.pop(0)
                destination = output / f"budget-{int(budget)}"
                destination.mkdir()
                torch.save(best_state, destination / "state_dict.pt")
                score = public_score(model)
                (destination / "score.json").write_text(
                    json.dumps(score, indent=2, allow_nan=False) + "\n"
                )
                saved.append(dict(budget_seconds=budget, fit_seconds=checkpoint_fit_seconds))
            model.load_state_dict(current)
            excluded_scoring_seconds += time.monotonic() - scoring_start
    else:
        update = updates
    model.load_state_dict(best_state)
    fitted_seconds = elapsed()
    torch.save(best_state, output / "state_dict.pt")
    score = public_score(model)
    (output / "public.json").write_text(json.dumps(score, indent=2, allow_nan=False) + "\n")
    result = dict(
        completed=True,
        best_training_loss=best_loss,
        loss_evaluations=evaluations,
        completed_updates=update if stop == "max_updates" else update - 1,
        stop=stop,
        fit_seconds=fitted_seconds,
        scoring_seconds=excluded_scoring_seconds,
        total_seconds=time.monotonic() - started,
        checkpoints=saved,
        initial_seed=int(config.get("model", {}).get("seed", 0)),
        training_config=config,
        public=score,
        trace=trace,
    )
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("completed", "stop", "fit_seconds", "loss_evaluations")}
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=300)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--checkpoints", type=float, nargs="*", default=[])
    args = parser.parse_args()
    fit(args.application, args.output, args.wall_seconds, args.max_updates, args.checkpoints)
