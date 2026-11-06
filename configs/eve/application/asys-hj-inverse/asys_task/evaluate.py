"""Evaluator-owned fitting and fixed held-out equation/constraint scoring."""

import argparse
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import torch
import yaml


class FitDeadline(Exception):
    pass


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_sources(case):
    for name in ("model.py", "loss.py", "optim.py"):
        tree = ast.parse((case / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for module in modules:
                    if module.split(".")[0] not in {
                        "math",
                        "torch",
                        "problems",
                        "__future__",
                    }:
                        raise ValueError(
                            f"{name}: candidate import is not a mathematical dependency: {module}"
                        )
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in {
                    "open",
                    "eval",
                    "exec",
                    "__import__",
                    "compile",
                }:
                    raise ValueError(
                        f"{name}: candidate file/code access is prohibited"
                    )
                if isinstance(func, ast.Attribute) and func.attr in {
                    "load",
                    "save",
                    "loadtxt",
                    "from_file",
                    "solve_ivp",
                    "odeint",
                }:
                    raise ValueError(
                        f"{name}: external state or numerical solution construction is prohibited"
                    )


def validate_model(model, sample):
    banned = (
        torch.nn.Linear,
        torch.nn.modules.conv._ConvNd,
        torch.nn.RNNBase,
        torch.nn.MultiheadAttention,
        torch.nn.Transformer,
    )
    if any(isinstance(layer, banned) for layer in model.modules()):
        raise ValueError("Neural solution layers are prohibited")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Candidate needs trainable mathematical parameters")
    value = model(sample)
    if value.ndim != 2 or len(value) != len(sample) or not torch.isfinite(value).all():
        raise ValueError(
            "Candidate must produce a finite matrix with one row per input"
        )
    gradients = torch.autograd.grad(value.square().sum(), parameters, allow_unused=True)
    if not any(
        g is not None and torch.isfinite(g).all() and torch.any(g != 0)
        for g in gradients
    ):
        raise ValueError("Candidate output must depend on its trainable parameters")


def evaluate(case_file):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save-state-dict")
    args = parser.parse_args()
    case = Path(case_file).resolve().parent
    config = yaml.safe_load((case / "config.yaml").read_text()) or {}
    for key, variable in (("Nf", "ASYS_NF"), ("NIC", "ASYS_NIC"), ("NBC", "ASYS_NBC")):
        if variable in os.environ:
            config[key] = int(os.environ[variable])
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    validate_sources(case)
    protected = load_module(case / "problem.py", "fixed_problem")
    helpers = protected.Problem(os.environ.get("ASYS_DEVICE", "cpu"))
    if "ASYS_NX" in os.environ:
        for field in ("radial_cells", "spatial_nodes"):
            if hasattr(helpers, field):
                setattr(helpers, field, int(os.environ["ASYS_NX"]))
    candidate = (
        load_module(case / "model.py", "candidate_model")
        .Candidate(config)
        .to(helpers.device, dtype=torch.float64)
    )
    batch = helpers.training_batch(config)
    validate_model(candidate, batch[0][:8])
    loss_fn = load_module(case / "loss.py", "candidate_loss").loss_fn
    optimizer = load_module(case / "optim.py", "candidate_optim").build_stepper(
        list(candidate.parameters()), helpers
    )
    budget = min(
        helpers.budget_seconds,
        float(os.environ.get("ASYS_WALL_TIME", helpers.budget_seconds)),
    )
    patience = float(os.environ.get("ASYS_EARLYSTOP_PATIENCE_FRAC", "0.3"))
    deadline = time.monotonic() + max(0, budget)
    best = math.inf
    best_state = {
        key: value.detach().clone() for key, value in candidate.state_dict().items()
    }
    history = []
    started = time.monotonic()
    last_improvement = started

    def closure():
        nonlocal best, best_state, last_improvement
        if time.monotonic() >= deadline:
            raise FitDeadline()
        optimizer.zero_grad()
        loss = loss_fn(candidate, *batch, helpers)
        if loss.ndim or not torch.isfinite(loss):
            raise ValueError("Training loss must be a finite scalar")
        loss.backward()
        if not all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in candidate.parameters()
        ):
            raise ValueError("Non-finite training gradient")
        value = float(loss.detach())
        history.append(value)
        if value < best:
            best = value
            last_improvement = time.monotonic()
            best_state = {
                key: tensor.detach().clone()
                for key, tensor in candidate.state_dict().items()
            }
        return loss

    try:
        for _ in range(int(config.get("max_steps", 100000))):
            optimizer.step(closure)
            now = time.monotonic()
            if (
                now - started >= 0.15 * budget
                and now - last_improvement >= patience * budget
            ):
                break
    except FitDeadline:
        pass
    candidate.load_state_dict(best_state)
    # Candidate fitting settings never enter the monitor helper or its samples.
    monitor = protected.Problem(str(helpers.device))
    payload = monitor.score(candidate)
    payload["diagnostics"] = monitor.diagnostics(candidate)
    payload["score_components"] = {
        "score_n_collocation": len(monitor.batch(seed=10007)[0]),
        "train_n_collocation": len(batch[0]),
    }
    payload["summary"] = case.name + ": fixed equation and constraint losses"
    if args.save_state_dict:
        destination = Path(args.save_state_dict)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(candidate.state_dict(), destination)
    if os.environ.get("ASYS_LOSS_HISTORY"):
        import numpy as np

        np.savez(os.environ["ASYS_LOSS_HISTORY"], loss=history)
    if os.environ.get("ASYS_EVAL_REPORT"):
        Path(os.environ["ASYS_EVAL_REPORT"]).write_text(
            json.dumps(payload, indent=2) + "\n"
        )
    print(json.dumps(payload, allow_nan=False))
