"""Fit one continuous transient field in bulk, core, and wall coordinates."""
import torch


def loss_fn(model, collocation, ic_points, bc_points, helpers):
    # Logarithmic core coverage is geometric sampling, not a law for the clock.
    core = helpers.core_points(192, seed=62037)
    # A heat corner has distance O(sqrt(t)).  Transform the bulk training
    # points to cover that layer as well as the whole physical interval.
    wall_source = collocation[:min(64, len(collocation))]
    wall_time = wall_source[:, 1:]
    wall_radius = 5 - 6*(wall_source[:, :1]/5)*wall_time.sqrt()
    wall = torch.cat((wall_radius, wall_time), dim=1)
    points = torch.cat((collocation, core, wall, bc_points), dim=0)
    residual, scale = helpers.pde_terms(model, points)
    squared = (residual/scale).square()
    bulk_loss, core_loss, wall_loss, origin_loss = torch.split(
        squared, [len(collocation), len(core), len(wall), len(bc_points)])
    # IC, interior initial tangent and both boundaries hold by construction.
    # A small mode penalty selects smooth corrections when modes are redundant.
    regularity = model.outer_modes.square().mean() + model.core_modes.square().mean()
    return (bulk_loss.mean() + 0.5*core_loss.mean()
            + 0.25*wall_loss.mean() + 0.25*origin_loss.mean() + 1e-9*regularity)
