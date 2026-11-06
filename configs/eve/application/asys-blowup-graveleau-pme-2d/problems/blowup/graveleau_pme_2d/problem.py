import math
import torch
from asys_task.problem import EvolutionProblem
from asys_task.operators import derivative, geometric_relative


class Problem(EvolutionProblem):
    T_FOCUS_APPROX = 0.6200901156700502
    T_MAX = 0.995 * T_FOCUS_APPROX
    R_MAX = 6
    R_EPS = 0.002
    HOLE_RADIUS = 1
    M_EXPONENT = 2
    N_DIM = 2
    bounds = ((R_EPS, R_MAX), (0, T_MAX))
    dt = 6.1699e-4

    def batch(self, count=None, ic_count=None, boundary_count=None, seed=0):
        points, initial, boundary = super().batch(count, ic_count, boundary_count, seed)
        # Fixed mixture covers the outer shell and resolves the shrinking inner front.
        # It uses the public initial hole, without a fitted front or reference trajectory.
        biased = self.uniform(len(points) // 2, seed + 3)
        unit = (biased[:, 0] - self.R_EPS) / (self.R_MAX - self.R_EPS)
        biased[:, 0] = self.R_EPS * (self.HOLE_RADIUS / self.R_EPS) ** unit
        points[: len(biased)] = biased
        return points, initial, boundary

    @staticmethod
    def initial_condition_torch(radius):
        distance = (radius - 1).clamp_min(0)
        return 1.6 * (1 - torch.exp(-(distance / 0.6).square()))

    initial_condition = initial_condition_torch

    def initial(self, points):
        return self.initial_condition_torch(points[:, :1])

    @staticmethod
    def graveleau_rhs(value, radial_derivative, radial_laplacian):
        return value * radial_laplacian + radial_derivative.square()

    def rhs(self, value, points):
        vr = derivative(value, points, 0)
        return self.graveleau_rhs(
            value, vr, derivative(vr, points, 0) + vr / points[:, :1]
        )

    def radial_derivative(self, model, points):
        points = points.detach().clone().requires_grad_(True)
        return derivative(self.as_scalar_field(model(points)), points, 0)

    def radial_laplacian(self, model, points):
        points = points.detach().clone().requires_grad_(True)
        vr = derivative(self.as_scalar_field(model(points)), points, 0)
        return derivative(vr, points, 0) + vr / points[:, :1]

    def pde_residual(self, model, points):
        points = points.detach().clone().requires_grad_(True)
        value = self.as_scalar_field(model(points))
        return derivative(value, points, 1) - self.rhs(value, points)

    def residual(self, model, points):
        return self.pde_residual(model, points), ()

    def outer_neumann_bc(self, model, points):
        """Return the outer Neumann residual, v_r(R_MAX,t), at the supplied times."""
        boundary = points.detach().clone()
        boundary[:, 0] = self.R_MAX
        return self.radial_derivative(model, boundary)

    def radial_boundary(self, model, left, right, a, b):
        return self.outer_neumann_bc(model, right).square().mean()

    @staticmethod
    def pressure_to_density(value, m=2):
        return ((m - 1) * value / m).clamp_min(0) ** (1 / (m - 1))

    def diagnostics(self, model):
        points = self.batch(seed=21007)[0].requires_grad_(True)
        pressure = model(points)
        vr = derivative(pressure, points, 0)
        time = derivative(pressure, points, 1)
        diffusion = pressure * (derivative(vr, points, 0) + vr / points[:, :1])
        gradient = vr.square()
        relative = geometric_relative(
            time - diffusion - gradient, (time, diffusion, gradient)
        )
        with torch.no_grad():
            radius = torch.linspace(self.R_EPS, self.R_MAX, 256, device=self.device)
            times = torch.linspace(0, self.T_MAX, 9, device=self.device)
            grid = torch.stack(
                [radius.repeat(len(times)), times.repeat_interleave(len(radius))], dim=1
            )
            field = model(grid).reshape(len(times), -1)
            density = self.pressure_to_density(field)
            mass = 2 * math.pi * torch.trapezoid(density * radius, radius, dim=-1)
            positive = field > 1e-6
            front_index = positive.long().argmax(dim=-1)
            front = torch.where(positive.any(dim=-1), radius[front_index], self.R_MAX)
            return {
                "dimensionless_residual": float(relative.detach()),
                "pressure_min": float(field.min()),
                "hole_pressure_mse": float(field[:, 0].square().mean()),
                "resolved_front_radius_min": float(front.min()),
                "resolved_front_radius_max": float(front.max()),
                "density_mass_min": float(mass.min()),
                "density_mass_max": float(mass.max()),
            }
