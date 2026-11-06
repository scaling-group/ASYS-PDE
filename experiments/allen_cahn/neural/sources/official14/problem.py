import torch
from asys_task.problem import EvolutionProblem
from asys_task.operators import derivative


def initial_condition(points):
    x, y = points[:, :1], points[:, 1:2]
    phases = []
    for cx, cy, radius in ((-0.15, 0.02, 0.22), (0.18, -0.03, 0.18)):
        distance = torch.sqrt((x - cx).square() + (y - cy).square() + 1e-12)
        phases.append(torch.tanh((radius - distance) / (2**0.5 * 0.04)))
    a, b = phases
    return (a + b + 1 - a * b) / 2


class Problem(EvolutionProblem):
    bounds = ((-1, 1), (-1, 1), (0, 8))
    periodic = True
    dt = 0.008
    ic_count = 1024

    def initial(self, points):
        return initial_condition(points)

    def rhs(self, value, points):
        laplacian = sum(
            derivative(derivative(value, points, axis), points, axis) for axis in (0, 1)
        )
        return 0.04**2 * laplacian + value - value**3

    def residual(self, model, points):
        value = model(points)
        return derivative(value, points, 2) - self.rhs(value, points), ()
