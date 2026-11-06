"""Editable scalar training objective; public scores are computed separately."""


def loss_fn(model, collocation, ic_points, bc_points, helpers):
    terms = helpers.loss_terms(model, collocation, ic_points, bc_points)
    return sum(terms.values())
