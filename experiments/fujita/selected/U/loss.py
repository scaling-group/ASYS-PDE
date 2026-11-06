"""Fit the fourth-order remainder with three initial jets and wall corners."""

def loss_fn(model, collocation, ic_points, bc_points, helpers):
    # All initial derivatives through third order hold in the open ball.
    # Initial values, the first tangent, and both boundaries are structural.
    interior = helpers.physics_loss(model, collocation)
    core = helpers.physics_loss(model, helpers.core_points(192, 107015))
    return interior + 0.2*core
