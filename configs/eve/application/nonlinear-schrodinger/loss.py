def loss_fn(model, collocation, ic_points, bc_points, helpers):
    return (helpers.residual(model, collocation).square().mean()
            + 15*(helpers.ic_loss(model, ic_points)+helpers.boundary_loss(model, bc_points))
            + helpers.tangent_loss(model, ic_points))
