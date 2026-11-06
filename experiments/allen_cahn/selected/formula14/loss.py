"""Fit the continuous geometry, with extra resolution around the peanut."""


def loss_fn(model, collocation, ic_points, bc_points, helpers):
    terms = helpers.loss_terms(model, collocation, ic_points, bc_points)
    focused = collocation.clone()
    focused[:, :2] = 0.55 * focused[:, :2]
    # O(1e-4) unscaled losses make LBFGS curvature pairs too small for its
    # absolute acceptance threshold. Rescale uniformly, preserving all weights.
    return 1000 * (sum(terms.values()) + 0.35 * helpers.physics_loss(model, focused))
