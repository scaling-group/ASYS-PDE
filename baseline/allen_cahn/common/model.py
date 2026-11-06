"""Fresh neural control with exact initial trace/jet and a bounded field.

The initial logit is the algebraic two-disk union. Its generator, positive-time
core smoothing and short-time damping are PDE priors, not fitted answers.
"""
import math

import torch
from torch import nn

from problem import EPS


class Candidate(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        width = 3
        for _ in range(4):
            layers.extend((nn.Linear(width, 64), nn.Tanh()))
            width = 64
        layers.append(nn.Linear(width, 1))
        self.correction = nn.Sequential(*layers)
        for layer in self.correction:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, points):
        xy, t = points[:, :2], points[:, 2:3]
        width = math.sqrt(2.0) * EPS
        delta = xy[:, None, :] - xy.new_tensor([[-0.15, 0.02], [0.18, -0.03]])
        radius = (delta.square().sum(2) + 1.0e-14).sqrt()
        radii = xy.new_tensor([0.22, 0.18])
        a = 2.0 * (radii - radius) / width
        branches = torch.cat((a, a.sum(1, keepdim=True)), dim=1)
        weights = torch.softmax(branches, dim=1)
        h0 = torch.logsumexp(branches, dim=1, keepdim=True) / 2.0
        normals = delta / radius[:, :, None]
        branch_normals = torch.cat((normals, normals.sum(1, keepdim=True)), dim=1)
        mean_normal = (weights[:, :, None] * branch_normals).sum(1)
        mean_squared = mean_normal.square().sum(1, keepdim=True)
        variance = (weights * branch_normals.square().sum(2)).sum(1, keepdim=True) - mean_squared
        radial_laplacian = 1.0 / radius + 1.0e-14 / radius.pow(3)
        lobe_weights = weights[:, :2] + weights[:, 2:3]
        velocity = (-EPS ** 2 / width * (lobe_weights * radial_laplacian).sum(1, keepdim=True)
                    + variance + h0.tanh() * (1.0 - mean_squared))
        # Smooth the radial cores without moving their individual zero circles.
        # Subtract this map's initial velocity so the jet is not double counted.
        smoothing = 2.0 * EPS ** 2 * (-torch.expm1(-t))
        smooth_radius = (radius.square() + smoothing).sqrt()
        smooth_radii = (radii.square() + smoothing).sqrt()
        geometry_a = 2.0 * (smooth_radii - smooth_radius) / width
        geometry = torch.logsumexp(torch.cat((geometry_a, geometry_a.sum(1, keepdim=True)), dim=1),
                                   dim=1, keepdim=True) / 2.0
        geometric_velocity = EPS ** 2 / width * (
            lobe_weights * (radii.reciprocal() - radius.reciprocal())).sum(1, keepdim=True)
        damping = 1.0 + EPS ** 2 * radius.reciprocal().square().sum(1, keepdim=True)
        jet = t * torch.exp(-t * damping)
        normalized = points * points.new_tensor([1.0, 1.0, 0.25])
        normalized = normalized + points.new_tensor([0.0, 0.0, -1.0])
        return torch.tanh(geometry + jet * (velocity - geometric_velocity)
                          + (t / 8.0).square() * self.correction(normalized))
