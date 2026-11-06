"""Protected radial n=5 cubic-reaction IVP; no solution/reference data."""
import math
import numpy as np
import torch
from asys_task.problem import EvolutionProblem
from asys_task.operators import derivative


class Problem(EvolutionProblem):
    R_MAX = 5.0
    T_MAX = 0.020385594
    HORIZON_CLOCK = 0.020406  # Sampling metadata, not a certified blow-up time.
    bounds = ((0, R_MAX), (0, T_MAX))
    physics_count, ic_count, boundary_count = 2048, 512, 256
    dimensions = ("physics", "ic", "ic_tangent", "bc")

    @staticmethod
    def lhs(count, dimensions, rng):
        values = np.column_stack([(rng.permutation(count) + rng.random(count)) / count
                                  for _ in range(dimensions)])
        return np.clip(values, np.nextafter(0., 1.), np.nextafter(1., 0.))

    def batch(self, count=None, ic_count=None, boundary_count=None, seed=920614):
        rng = np.random.default_rng(seed)
        n, ni, nb = count or self.physics_count, ic_count or self.ic_count, boundary_count or self.boundary_count
        physics = self.lhs(n, 2, rng) * [self.R_MAX, self.T_MAX]
        radius = np.r_[0., self.R_MAX * self.lhs(ni-2, 1, rng)[:, 0], self.R_MAX]
        initial = np.c_[radius, np.zeros(ni)]
        times = np.r_[0., self.T_MAX * self.lhs(nb-2, 1, rng)[:, 0], self.T_MAX]
        boundary = np.c_[np.zeros(nb), times]
        return tuple(torch.tensor(x, dtype=torch.float64, device=self.device)
                     for x in (physics, initial, boundary))

    def training_batch(self, config=None):
        cfg = config or {}
        return self.batch(count=int(cfg.get("Nf", 256)),
                          ic_count=int(cfg.get("NIC", 64)),
                          boundary_count=int(cfg.get("NBC", 32)), seed=107014)

    def core_points(self, count, seed):
        rng = np.random.default_rng(seed)
        values = self.lhs(count, 2, rng)
        tau = self.HORIZON_CLOCK * np.exp(math.log(.001) + math.log(100) * values[:, 1])
        interior = np.c_[10*np.sqrt(tau)*values[:, 0], self.HORIZON_CLOCK-tau]
        origin_tau = self.HORIZON_CLOCK * np.exp(rng.uniform(math.log(.001), math.log(.1), max(16, count//16)))
        origin = np.c_[np.zeros(len(origin_tau)), self.HORIZON_CLOCK-origin_tau]
        return torch.tensor(np.r_[interior, origin], dtype=torch.float64, device=self.device)

    def initial(self, points):
        return 5*torch.cos(math.pi*points[:, :1]/10)

    def initial_rhs(self, points):
        radius = points[:, :1]
        k = math.pi/10
        # torch.sinc supplies the regular origin value without dividing by r.
        return -5*k*k*torch.cos(k*radius) - 20*k*k*torch.sinc(k*radius/math.pi) + self.initial(points).pow(3)

    def pde_terms(self, model, points):
        points = points.detach().clone().requires_grad_(True)
        value = self.as_scalar_field(model(points))
        radial = derivative(value, points, 0)
        second = derivative(radial, points, 0)
        radius = points[:, :1]
        safe = torch.where(radius == 0, torch.ones_like(radius), radius)
        diffusion = second + 4*torch.where(radius == 0, second, radial/safe)
        temporal, reaction = derivative(value, points, 1), value.pow(3)
        scale = (temporal.abs()+diffusion.abs()+reaction.abs()).clamp_min(1e-12)
        return temporal-diffusion-reaction, scale

    def residual(self, model, points):
        return self.pde_terms(model, points)[0], ()

    def physics_loss(self, model, points):
        residual, scale = self.pde_terms(model, points)
        return (residual/scale).square().mean()

    def radial_boundary(self, model, left, right, a, b):
        return (derivative(a, left, 0).square().mean()+b.square().mean())/2

    def loss_terms(self, model, collocation, ic_points, bc_points):
        # The outer initial corner is incompatible with a fixed Dirichlet wall.
        interior = ic_points[ic_points[:, 0] < self.R_MAX].detach().clone().requires_grad_(True)
        tangent = derivative(model(interior), interior, 1)
        return {
            "physics": self.physics_loss(model, collocation),
            "ic": (model(ic_points)-self.initial(ic_points)).square().mean(),
            "ic_tangent": (tangent-self.initial_rhs(interior)).square().mean(),
            "bc": self.boundary_loss(model, bc_points),
        }

    def score(self, model):
        batch = self.batch(seed=920614)
        losses = self.loss_terms(model, *batch)
        residual, scale = self.pde_terms(model, batch[0])
        losses["physics"] = (residual.abs()/scale).clamp_min(1e-12).log().mean().exp()
        scores = {name: -float(losses[name].detach()) for name in self.dimensions}
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError("Nonfinite Fujita score")
        return {"score": sum(scores.values())/len(scores), **scores}

    def diagnostics(self, model):
        points = self.core_points(2048, 920811)
        values = model(points)
        return {"core_normalized_residual_mse": float(self.physics_loss(model, points).detach()),
                "sampled_core_min": float(values.min().detach()),
                "sampled_core_max": float(values.max().detach())}
