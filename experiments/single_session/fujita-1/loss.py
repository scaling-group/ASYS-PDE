"""Fit the continuous field in the outer region, core, and heat corner."""
import torch


def loss_fn(model, collocation, ic_points, bc_points, helpers):
    # Independent training core samples, including regular-origin points.
    count = min(192, max(64, len(collocation)//2))
    core = helpers.core_points(count, seed=107115)
    # Diffusion determines the corner's physical length scale sqrt(t).
    samples = collocation[:min(96, len(collocation))]
    t = samples[:, 1:]
    distance = 4*torch.sqrt(t)*samples[:, :1]/5
    corner = torch.cat((5-distance, t), dim=1)
    origin = torch.cat((torch.zeros_like(t[:32]), t[:32]), dim=1)
    # Initial field/tangent and boundaries are built into the expression.
    # The tiny moment penalty regularizes the truncated analytic expansion.
    moments = (model.outer_moments.square().mean()
               + model.core_moments.square().mean())
    return (helpers.physics_loss(model, collocation)
            + 0.5*helpers.physics_loss(model, core)
            + 0.25*helpers.physics_loss(model, corner)
            + 0.1*helpers.physics_loss(model, origin)
            + 1e-9*moments)
