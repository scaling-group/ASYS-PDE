"""Shared equation/constraint measurement for pointwise formula programs."""

import math
import torch
from .operators import DTYPE, derivative, geometric_relative


class EvolutionProblem:
    dimensions = ("physics", "ic", "ic_tangent", "bc")
    relative_physics = False
    periodic = False
    components = 1
    budget_seconds = 300
    ic_count = 256
    physics_count = 1024
    boundary_count = 128

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    @staticmethod
    def as_scalar_field(value):
        if value.ndim == 1:
            return value[:, None]
        if value.ndim == 2 and value.shape[1] == 1:
            return value
        raise ValueError("Expected one scalar field value per point")

    def diagnostics(self, model):
        return {}

    def uniform(self, count, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        unit = torch.rand(
            count,
            len(self.bounds),
            dtype=DTYPE,
            device=self.device,
            generator=generator,
        )
        lower = unit.new_tensor([lo for lo, _ in self.bounds])
        scale = unit.new_tensor([hi - lo for lo, hi in self.bounds])
        return lower + scale * unit

    def batch(self, count=None, ic_count=None, boundary_count=None, seed=0):
        interior = self.uniform(count or self.physics_count, seed)
        initial = self.uniform(ic_count or self.ic_count, seed + 1)
        initial[:, -1] = 0
        boundary = self.uniform(boundary_count or self.boundary_count, seed + 2)
        return interior, initial, boundary

    def boundary_loss(self, model, points):
        terms = []
        for axis, (lo, hi) in enumerate(self.bounds[:-1]):
            left, right = points.clone(), points.clone()
            left[:, axis], right[:, axis] = lo, hi
            left.requires_grad_(True)
            right.requires_grad_(True)
            a, b = model(left), model(right)
            if self.periodic:
                terms.append((a - b).square().mean())
                for channel in range(self.components):
                    terms.append(
                        (
                            derivative(a[:, channel : channel + 1], left, axis)
                            - derivative(b[:, channel : channel + 1], right, axis)
                        )
                        .square()
                        .mean()
                    )
            else:
                terms.append(self.radial_boundary(model, left, right, a, b))
        return torch.stack(terms).mean()

    def physics_loss(self, model, points):
        points = points.detach().clone().requires_grad_(True)
        residual, terms = self.residual(model, points)
        if self.relative_physics:
            return geometric_relative(residual, terms)
        return residual.square().mean()

    def loss_terms(self, model, collocation, ic_points, bc_points):
        initial = self.initial(ic_points)
        losses = {
            "physics": self.physics_loss(model, collocation),
            "ic": (model(ic_points) - initial).square().mean(),
            "bc": self.boundary_loss(model, bc_points),
        }
        advanced = ic_points.clone()
        advanced[:, -1] = self.dt
        tangent = (model(advanced) - model(ic_points)) / self.dt
        losses["ic_tangent"] = (tangent - self.initial_rhs(ic_points)).square().mean()
        return losses

    def training_batch(self, config=None):
        cfg = config or {}
        return self.batch(
            count=int(cfg.get("Nf", self.physics_count)),
            ic_count=int(cfg.get("NIC", self.ic_count)),
            boundary_count=int(cfg.get("NBC", self.boundary_count)),
            seed=11,
        )

    def score(self, model):
        losses = self.loss_terms(model, *self.batch(seed=10007))
        result = {name: -float(losses[name].detach()) for name in self.dimensions}
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("Non-finite equation or constraint score")
        return {"score": sum(result.values()) / len(result), **result}

    def initial_rhs(self, points):
        points = points.detach().clone().requires_grad_(True)
        value = self.initial(points)
        return self.rhs(value, points).detach()
