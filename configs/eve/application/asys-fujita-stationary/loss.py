def loss_fn(model, collocation, ic_points, bc_points, helpers):
    terms = helpers["components"](model, collocation, ic_points, bc_points)
    return sum(terms.values())
