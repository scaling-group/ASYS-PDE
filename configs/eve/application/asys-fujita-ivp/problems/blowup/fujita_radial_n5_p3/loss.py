"""Editable normalized PDE/initial/boundary objective; scores are protected."""

def loss_fn(model, collocation, ic_points, bc_points, helpers):
    terms = helpers.loss_terms(model, collocation, ic_points, bc_points)
    return terms["physics"] + terms["ic"] + terms["bc"]
