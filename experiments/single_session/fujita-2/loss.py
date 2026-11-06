"""Fit the same transient field throughout the domain and concentrating core."""

def loss_fn(model, collocation, ic_points, bc_points, helpers):
    # Initial value, interior initial tangent, and both boundaries are built
    # into the formula.  Fit only the continuous PDE, with extra resolution
    # in the late core using the public geometric coverage convention.
    core = helpers.core_points(max(128, len(collocation)//2), seed=107035)
    return (helpers.physics_loss(model, collocation)
            +helpers.physics_loss(model, core))
