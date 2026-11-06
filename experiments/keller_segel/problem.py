"""Fixed KS mathematical API and local ASYS numerical fitting implementation.

Selected from the current task at ASYS7283bd22. The fitting function is retained
verbatim; unrelated PDE helpers and the old evaluation entrypoint are omitted.
"""
from __future__ import annotations
import copy
import importlib.util
import inspect
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
import numpy as np
import torch
DTYPE = torch.float64
PUBLIC_CELLS = 1280


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

def _scalar_field(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values[:, None]
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(f"expected one output column, got {tuple(values.shape)}")
    return values

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

class BaseHelpers:
    name = "base"

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.DTYPE = DTYPE
        for key, value in config.items():
            if isinstance(value, (int, float)):
                setattr(self, key, value)

    @staticmethod
    def as_scalar_field(values: torch.Tensor) -> torch.Tensor:
        return _scalar_field(values)






class KSHelpers(BaseHelpers):
    name = "ks_radial_2d"
    R_MIN_FACE = 1.0e-5
    R_MAX = 30.0
    T_STAR_APPROX = 0.9628
    T_STAR = T_STAR_APPROX
    T_MAX = 0.9 * T_STAR_APPROX
    CRITICAL_MASS = 8.0 * math.pi
    MASS0 = 9.5 * math.pi

    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)
        n_r = int(config.get("n_x", 64))
        (
            self.COLLOCATION_R,
            self.COLLOCATION_R_FACE,
            self.COLLOCATION_CELL_MEASURE,
        ) = self._radial_grid(n_r)
        self.collocation_r = self.COLLOCATION_R
        self.collocation_r_face = self.COLLOCATION_R_FACE
        self.collocation_cell_measure = self.COLLOCATION_CELL_MEASURE

    def initial_condition(self, r: torch.Tensor) -> torch.Tensor:
        return 9.5 * torch.exp(-r.square())

    def _radial_grid(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cell centers, faces, and exact radial cell measures."""
        r_face = torch.zeros(n + 1, dtype=DTYPE, device=self.device)
        r_face[1:] = torch.logspace(
            math.log10(self.R_MIN_FACE),
            math.log10(self.R_MAX),
            n,
            dtype=DTYPE,
            device=self.device,
        )
        r = 0.5 * (r_face[:-1] + r_face[1:])
        cell_measure = 0.5 * (r_face[1:].square() - r_face[:-1].square())
        return r, r_face, cell_measure

    @staticmethod
    def ks_M_cumsum(values: torch.Tensor, cell_measure: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(values * cell_measure, dim=-1)

    @staticmethod
    def ks_mass(values: torch.Tensor, cell_measure: torch.Tensor) -> torch.Tensor:
        return 2.0 * math.pi * torch.sum(values * cell_measure, dim=-1)

    def _ks_operator_parts(
        self,
        values: torch.Tensor,
        r: torch.Tensor,
        r_face: torch.Tensor,
        cell_measure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Conservative cell-centered radial finite-volume operator."""
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != r.numel():
            raise ValueError("values must have shape [n_time, n_radial]")
        if r_face.numel() != r.numel() + 1:
            raise ValueError("r_face must contain one more entry than r")

        n_time, n_r = values.shape
        mass_face = values.new_zeros((n_time, n_r + 1))
        mass_face[:, 1:] = torch.cumsum(values * cell_measure[None, :], dim=1)

        gradient_face = values.new_zeros((n_time, n_r + 1))
        gradient_face[:, 1:n_r] = (values[:, 1:] - values[:, :-1]) / (r[1:] - r[:-1])[None, :]
        # Origin symmetry gives zero flux.  At the remote outer boundary,
        # linear decay to zero is consistent with the separately scored
        # value/derivative decay conditions.
        gradient_face[:, -1] = -values[:, -1] / (r_face[-1] - r[-1])

        value_face = values.new_zeros((n_time, n_r + 1))
        value_face[:, 1:n_r] = 0.5 * (values[:, :-1] + values[:, 1:])

        diffusion_flux = r_face[None, :] * gradient_face
        chemistry_flux = value_face * mass_face
        diffusion = (diffusion_flux[:, 1:] - diffusion_flux[:, :-1]) / cell_measure[None, :]
        chemistry = (chemistry_flux[:, 1:] - chemistry_flux[:, :-1]) / cell_measure[None, :]
        return diffusion + chemistry, diffusion, chemistry, mass_face[:, 1:]

    def ks_rhs_on_grid(
        self,
        values: torch.Tensor,
        r: torch.Tensor,
        r_face: torch.Tensor,
        cell_measure: torch.Tensor,
    ) -> torch.Tensor:
        return self._ks_operator_parts(values, r, r_face, cell_measure)[0]

    def _grid(
        self,
        n_r: int,
        n_t: int,
        *,
        times: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r, r_face, cell_measure = self._radial_grid(n_r)
        t = (
            torch.linspace(0.0, self.T_MAX, n_t, dtype=DTYPE, device=self.device)
            if times is None
            else times.to(device=self.device, dtype=DTYPE)
        )
        if t.numel() != n_t:
            raise ValueError("times length must equal n_t")
        tt, rr = torch.meshgrid(t, r, indexing="ij")
        points = torch.stack([rr.reshape(-1), tt.reshape(-1)], dim=1)
        return points, r, r_face, cell_measure

    def _monitor_times(self) -> torch.Tensor:
        early = torch.linspace(0.0, 0.5 * self.T_MAX, 21, dtype=DTYPE, device=self.device)
        ratio = torch.tensor(0.89257, dtype=DTYPE, device=self.device)
        intervals = ratio.pow(torch.arange(15, dtype=DTYPE, device=self.device))
        intervals = 0.5 * self.T_MAX * intervals / intervals.sum()
        late = 0.5 * self.T_MAX + torch.cumsum(intervals, dim=0)
        return torch.cat([early, late])

    def _residual_parts(
        self, model: torch.nn.Module, points: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        coords = _clone_points(points)
        values = _scalar_field(model(coords))
        grad = _grad(values, coords)
        r = torch.unique(coords[:, 0].detach(), sorted=True)
        t = torch.unique(coords[:, 1].detach(), sorted=True)
        nt, nr = t.numel(), r.numel()
        grid_values = values.reshape(nt, nr)
        ut = grad[:, 1:2].reshape(nt, nr)
        expected_r, r_face, cell_measure = self._radial_grid(nr)
        if not torch.allclose(r, expected_r, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("Keller-Segel residual requires the evaluator radial grid")
        rhs, diffusion, chemistry, mass = self._ks_operator_parts(
            grid_values, r, r_face, cell_measure
        )
        return (
            (ut - rhs).reshape(-1, 1),
            ut,
            diffusion,
            chemistry,
            mass,
            r,
            grid_values,
        )

    def ks_flux_residual(self, model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
        return self._residual_parts(model, points)[0]

    pde_residual = ks_flux_residual

    def boundary_loss(self, model: torch.nn.Module, bc_points: torch.Tensor) -> torch.Tensor:
        t = bc_points[:, -1:]
        origin = torch.cat([torch.zeros_like(t), t], dim=1).requires_grad_(True)
        outer = torch.cat([torch.full_like(t, self.R_MAX), t], dim=1).requires_grad_(True)
        u0 = _scalar_field(model(origin))
        ur0 = _grad(u0, origin)[:, 0:1]
        uo = _scalar_field(model(outer))
        uro = _grad(uo, outer)[:, 0:1]
        return ur0.square().mean() + uo.square().mean() + uro.square().mean()

    def positivity_penalty(self, model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
        return torch.relu(-_scalar_field(model(points))).square().mean()

    def mass_loss(self, model: torch.nn.Module, split: str = "train") -> torch.Tensor:
        _ = split
        points, r, _, cell_measure = self._grid(96, 8)
        values = _scalar_field(model(points)).reshape(8, 96)
        target = self.ks_mass(self.initial_condition(r), cell_measure)
        mass = self.ks_mass(values, cell_measure)
        rel = (mass - target) / target.abs().clamp_min(1e-12)
        return rel.square().mean()

    def training_batch(self) -> TrainingBatch:
        n = int(self.config.get("n_collocation", 2048))
        n_r = int(self.config.get("n_x", 64))
        n_t = max(3, math.ceil(n / n_r))
        collocation, r, _, _ = self._grid(n_r, n_t)
        n_ic = int(self.config.get("n_ic", 256))
        ri, _, _ = self._radial_grid(n_ic)
        ri = ri[:, None]
        ic = torch.cat([ri, torch.zeros_like(ri)], dim=1)
        n_bc = int(self.config.get("n_bc", 128))
        bc = torch.linspace(0.0, self.T_MAX, n_bc, dtype=DTYPE, device=self.device)[:, None]
        _ = r
        return TrainingBatch(collocation, ic, bc)


    def radial_mass(self, model, radii, faces, times):
        """Four-point Gaussian integration in each full/partial radial cell.

        Public readout only: the expanded differential equation needs mass
        values, not derivatives of this numerical quadrature graph.
        """
        nodes, weights = np.polynomial.legendre.leggauss(4)
        nodes = torch.tensor(nodes, dtype=DTYPE, device=self.device)
        weights = torch.tensor(weights, dtype=DTYPE, device=self.device)

        def integrate(left, right):
            half = (right - left) / 2
            points_r = (right + left)[:, None] / 2 + half[:, None] * nodes
            rr = points_r[None].expand(len(times), -1, -1)
            tt = times[:, None, None].expand_as(rr)
            points = torch.stack((rr.flatten(), tt.flatten()), dim=1)
            with torch.no_grad():
                values = _scalar_field(model(points)).reshape_as(rr)
            if not bool(torch.isfinite(values).all()) or float(values.min()) < -1e-10:
                raise ValueError("Invalid density at mass quadrature points")
            return (values * rr * weights).sum(2) * half

        full = integrate(faces[:-1], faces[1:])
        previous = torch.cat((full.new_zeros((len(times), 1)), full.cumsum(1)[:, :-1]), dim=1)
        return previous + integrate(faces[:-1], radii)

    def continuous_parts(self, model, nodes, times):
        """u_t and the continuous radial diffusion/chemotactic terms at r>0."""
        points, r, faces, measure = self._grid(nodes, len(times), times=times)
        points = points.detach().requires_grad_(True)
        values = _scalar_field(model(points))
        gradient = _grad(values, points)
        radial_second = _grad(gradient[:, :1], points)[:, 0]
        radial_first = gradient[:, 0]
        mass = self.radial_mass(model, r, faces, times)
        diffusion = (radial_second + radial_first / points[:, 0]).reshape(len(times), nodes)
        chemistry = (radial_first / points[:, 0]).reshape(len(times), nodes) * mass
        chemistry = chemistry + values.reshape(len(times), nodes).square()
        ut = gradient[:, 1].reshape(len(times), nodes)
        return ut, diffusion, chemistry, values.reshape(len(times), nodes), measure


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
