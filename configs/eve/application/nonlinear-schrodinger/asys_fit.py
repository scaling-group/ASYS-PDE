"""Numerical primitives."""
from __future__ import annotations
import copy, importlib.util, inspect, math, os, sys, time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable
import torch
DTYPE = torch.float64

class BudgetReached(RuntimeError):
    pass

def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def _as_uv(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"expected two output columns, got {tuple(values.shape)}")
    return values[:, 0:1], values[:, 1:2]

def _grad(value: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if not value.requires_grad:
        return torch.zeros_like(coords)
    result = torch.autograd.grad(
        value,
        coords,
        grad_outputs=torch.ones_like(value),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(coords) if result is None else result

def _clone_points(points: torch.Tensor) -> torch.Tensor:
    return points.detach().clone().requires_grad_(True)

def _finite(value: torch.Tensor, fallback: float = 1.0e12) -> torch.Tensor:
    return torch.nan_to_num(value, nan=fallback, posinf=fallback, neginf=-fallback)

@dataclass
class TrainingBatch:
    collocation: torch.Tensor
    ic_points: torch.Tensor
    bc_points: torch.Tensor

def _build_model(module: ModuleType, config: dict, device: torch.device) -> torch.nn.Module:
    candidate_type = module.Candidate
    try:
        signature = inspect.signature(candidate_type)
        model = candidate_type(config) if len(signature.parameters) else candidate_type()
    except (TypeError, ValueError):
        try:
            model = candidate_type(config)
        except TypeError:
            model = candidate_type()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Candidate must be a torch.nn.Module")
    model = model.to(device=device, dtype=DTYPE)
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Candidate must expose at least one trainable parameter")
    return model

def _default_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "LBFGS")).lower()
    lr = float(config.get("lr", 0.5))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr if lr > 0 else 1e-3)
    return torch.optim.LBFGS(
        model.parameters(),
        lr=lr if lr > 0 else 0.5,
        max_iter=1,
        max_eval=max(1, int(config.get("max_eval", 20))),
        history_size=max(1, int(config.get("history_size", 50))),
        line_search_fn="strong_wolfe",
    )

def _build_optimizer(
    module: ModuleType, model: torch.nn.Module, config: dict
) -> torch.optim.Optimizer:
    builder = getattr(module, "build_optimizer", None)
    if builder is None:
        return _default_optimizer(model, config)
    optimizer = builder(model, config)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("build_optimizer must return torch.optim.Optimizer")
    return optimizer

def _fit(
    model: torch.nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    batch: TrainingBatch,
    helpers: BaseHelpers,
    wall_time: float,
) -> tuple[list[float], float, int]:
    started = time.monotonic()
    reserve = min(30.0, max(1.0, 0.20 * wall_time))
    deadline = started + max(0.5, wall_time - reserve)
    patience_fraction = float(os.getenv("ASYS_EARLYSTOP_PATIENCE_FRAC", "0.3"))
    patience = max(2.0, patience_fraction * max(1.0, wall_time - reserve))
    best_loss = math.inf
    best_state: dict | None = None
    best_at = started
    history: list[float] = []
    step = 0
    max_steps = int(helpers.config.get("max_steps", 100000))

    def closure() -> torch.Tensor:
        nonlocal best_loss, best_state, best_at
        if time.monotonic() >= deadline:
            raise BudgetReached
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch.collocation, batch.ic_points, batch.bc_points, helpers)
        if not torch.is_tensor(loss) or loss.numel() != 1:
            raise TypeError("loss_fn must return one scalar tensor")
        loss = _finite(loss.reshape(()))
        value = float(loss.detach().cpu())
        history.append(value)
        if value < best_loss and math.isfinite(value):
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            best_at = time.monotonic()
        loss.backward()
        return loss

    while step < max_steps and time.monotonic() < deadline:
        if time.monotonic() - best_at >= patience:
            break
        try:
            if isinstance(optimizer, torch.optim.LBFGS):
                optimizer.step(closure)
            else:
                closure()
                optimizer.step()
        except BudgetReached:
            break
        step += 1

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_loss, step
