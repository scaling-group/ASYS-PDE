from problem import terms


def loss_fn(model, collocation, ic_points, bc_points, helpers):
    values = terms(model, collocation, ic_points, bc_points)
    return sum(values.values())
