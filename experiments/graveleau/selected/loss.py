"""Exterior PME fitting with analytic pressure differentiation and quadrature.

The pressure ansatz fixes the initial field and outer Neumann symmetry.
Only their parameter-dependent tangent continuation needs fitting. All
conservation terms remain spatial integrals of the same continuous field.
The exponential and reflected quotient use explicit elementary chain rules;
autodiff still differentiates the complete symbolic shell exponent.
"""

import math
import torch


def _pressure_jet(model, points, second_order=False, detach=True):
    """Differentiate exponents and compose the pressure's analytic jet.

    For H(f,g)=A*f*g/D, D=f+g-f*g, the unguarded derivatives are
    H_f=A*g^2/D^2, H_ff=-2*H_f*(1-g)/D, H_fg=2*A*f*g/D^3.
    The constant-denominator branch is included as well. Both reflected
    branches have independent physical coordinates. Integral callers retain
    their map graph while differentiating in these physical coordinates.
    """
    radius = torch.cat((points[:, :1], 12 - points[:, :1]), dim=1)
    time = points[:, 1:2].expand(-1, 2).clone()
    if detach:
        radius, time = radius.detach(), time.detach()
    radius.requires_grad_(True)
    time.requires_grad_(True)
    front, speed, beta = model.front(time)
    raw_distance = radius - front
    distance = torch.where(raw_distance >= 0, raw_distance, torch.zeros_like(raw_distance))
    exponent = model.shell_exponent(distance, time, front, speed, beta)
    radial, temporal = torch.autograd.grad(
        exponent.sum(), (radius, time), create_graph=True,
    )
    decay = torch.exp(-exponent)
    factors = -torch.expm1(-exponent)
    f, g = factors[:, :1], factors[:, 1:2]
    product = f * g
    raw_denominator = f + g - product
    denominator = raw_denominator.clamp_min(1e-15)
    active = raw_denominator >= 1e-15
    other = factors.flip(dims=(-1,))
    factor_partial = torch.where(
        active, 1.6 * (other / denominator).square(),
        1.6 * other / denominator,
    )
    exponent_partial = factor_partial * decay
    physical_radial = radial * radius.new_tensor((1.0, -1.0))
    pressure = 1.6 * product / denominator
    slope = (exponent_partial * physical_radial).sum(dim=-1, keepdim=True)
    time_derivative = (exponent_partial * temporal).sum(dim=-1, keepdim=True)
    if not second_order:
        return pressure, slope, time_derivative

    second = torch.autograd.grad(radial.sum(), radius, create_graph=True)[0]
    diagonal = -exponent_partial * (
        1 + 2 * active.to(points.dtype) * (1 - other) * decay / denominator
    )
    cross = torch.where(
        active, 3.2 * product / denominator.pow(3), 1.6 / denominator,
    ) * decay.prod(dim=-1, keepdim=True)
    curvature = (
        exponent_partial * second + diagonal * physical_radial.square()
    ).sum(dim=-1, keepdim=True) + 2 * cross * physical_radial.prod(dim=-1, keepdim=True)
    return pressure, slope, time_derivative, curvature


def _pde_residual(model, points):
    points = points.detach()
    pressure, slope, time_derivative, curvature = _pressure_jet(
        model, points, second_order=True,
    )
    return time_derivative - pressure * (
        curvature + slope / points[:, :1]
    ) - slope.square()


def _time_rule(points, extent, subdivisions=4):
    """Composite Gauss quadrature of the uniform physical-time mean.

    Three-point Gauss rules cover four intervals set by the initial shell
    diffusion time w^2/(4*A). The early transient and final approach to focus
    each receive a short interval; no sampled solution values set the nodes.
    Four panels per interval resolve the squared weak-balance integrands.
    These are simultaneous evaluations of one field, never evolution steps.
    """
    diffusion_time = 0.6**2 / (4 * 1.6)
    edges = points.new_tensor((
        0.0, diffusion_time, 3 * diffusion_time, extent - diffusion_time, extent,
    ))
    nodes = points.new_tensor((-math.sqrt(3 / 5), 0.0, math.sqrt(3 / 5)))
    weights = points.new_tensor((5 / 9, 8 / 9, 5 / 9))
    widths = edges[1:] - edges[:-1]
    unit_midpoint = (
        torch.arange(subdivisions, dtype=points.dtype, device=points.device) + 0.5
    ) / subdivisions
    half_width = widths / (2 * subdivisions)
    midpoint = edges[:-1, None] + widths[:, None] * unit_midpoint
    return (
        (midpoint[..., None] + half_width[:, None, None] * nodes).flatten(),
        (half_width[:, None, None] / extent * weights)
        .expand(-1, subdivisions, -1).flatten(),
    )


def _initial_mass(extent):
    """Closed integral of r*A*(1-exp(-((r-H)/w)^2)) on [H,extent]."""
    amplitude, hole, width = 1.6, 1.0, 0.6
    upper = (extent - hole) / width
    return amplitude * (
        (extent**2 - hole**2) / 2
        - width**2 / 2 * (1 - math.exp(-upper**2))
        - hole * width * math.sqrt(math.pi) / 2 * math.erf(upper)
    )


def _initial_rhs(radius):
    """Analytic pressure-PME tangent of the supplied quadratic Gaussian edge."""
    amplitude, inverse_width2 = 1.6, 1 / 0.6**2
    distance = (radius - 1).clamp_min(0)
    exponent = inverse_width2 * distance.square()
    decay = torch.exp(-exponent)
    complement = -torch.expm1(-exponent)
    return amplitude**2 * decay * (
        2 * inverse_width2 * complement * (1 + distance / radius)
        + 4 * inverse_width2**2 * distance.square() * (2 * decay - 1)
    )


def _exterior_rule(front, extent, panels=16):
    """Four-point Lobatto panels in two smooth coordinates, split at L/2.

    The inner map r=R+(L/2-R)*x^2 resolves the one-sided edge and has
    zero Jacobian at R. The outer map is linear. Keep both copies of L/2:
    their distinct Jacobians belong to their respective integration panels.
    All maps and weights retain their dependence on the mathematical front.
    Each panel integrates degree-five polynomials exactly. Shared endpoints
    also supply the annular flux, requiring 3*panels+1 nodes per coordinate.
    """
    offsets = front.new_tensor((0.0, (1 - 1 / math.sqrt(5)) / 2,
                               (1 + 1 / math.sqrt(5)) / 2))
    starts = torch.arange(panels, dtype=front.dtype, device=front.device)
    unit = torch.cat((
        ((starts[:, None] + offsets) / panels).flatten(), front.new_ones(1),
    ))
    split = extent / 2
    inner = front + (split - front) * unit.square()
    outer = split + (extent - split) * unit + front * 0
    radius = torch.cat((inner, outer), dim=-1)
    jacobian = torch.cat((
        2 * (split - front) * unit,
        torch.ones_like(outer) * (extent - split),
    ), dim=-1)
    weights = torch.ones_like(unit)
    weights[::3] = 2
    weights[0] = weights[-1] = 1
    weights[1::3] = weights[2::3] = 5
    weights = torch.cat((weights, weights)) / (12 * panels)
    return radius, jacobian * weights


def _flux_residuals(pressure, time_derivative, slope, radius, radial_weights):
    """Annulus-averaged PDE defect at Lobatto panel ends.

    This is a spatial prefix integral of a single continuous pressure field,
    never a time update. Integrating only the exterior removes the jump in
    v_t at the interface. The zero inner Jacobian removes endpoint ambiguity.
    Since v(R)=0, the numerator is integral_R^r s*PDE_residual ds.
    Dividing by (r^2-R^2)/2 gives its radial area average, including near
    the front. Panel ends lie strictly outside R, so no cutoff is needed.
    """
    nodes = radius.shape[-1] // 2
    integrand = radial_weights * time_derivative
    inner = integrand[:, :nodes]
    outer = integrand[:, nodes:]
    # A shared endpoint has Lobatto weight two in the full integral, and
    # weight one in either neighboring panel. Undo that factor locally.
    def panel_integrals(values):
        left = values[:, :-3:3].clone()
        right = values[:, 3::3].clone()
        left[:, 1:] = left[:, 1:] / 2
        right[:, :-1] = right[:, :-1] / 2
        return left + values[:, 1::3] + values[:, 2::3] + right

    panels = torch.cat((panel_integrals(inner), panel_integrals(outer)), dim=-1)
    accumulation = torch.cumsum(panels, dim=-1)
    edge_radius = torch.cat((radius[:, 3:nodes:3], radius[:, nodes + 3::3]), dim=-1)
    edge_pressure = torch.cat((pressure[:, 3:nodes:3], pressure[:, nodes + 3::3]), dim=-1)
    edge_slope = torch.cat((slope[:, 3:nodes:3], slope[:, nodes + 3::3]), dim=-1)
    flux = edge_radius * edge_pressure * edge_slope
    front = radius[:, :1]
    annulus_area = (edge_radius - front) * (edge_radius + front) / 2
    return (accumulation - flux) / annulus_area


def _moment_residuals(pressure, time_derivative, slope, radius, radial_weights, extent):
    """One integration by parts: integral r*(phi*v_t+phi_r*v*v_r)."""
    outer_x = (radius / extent).square()
    inner_extent = extent / 2
    inner_x = (radius / inner_extent).square()
    complement = (1 - inner_x).clamp_min(0)
    tests = torch.stack(((1 - outer_x).square(), complement.pow(3)), dim=-1)
    slopes = torch.stack((
        -4 * radius / extent**2 * (1 - outer_x),
        -6 * radius / inner_extent**2 * complement.square(),
    ), dim=-1)
    normalizations = radius.new_tensor((extent**2 / 6, inner_extent**2 / 8))
    return (
        radial_weights[..., None] * (
            tests * time_derivative[..., None]
            + slopes * (pressure * slope)[..., None]
        )
    ).sum(dim=1) / normalizations


def _energy_residuals(pressure, time_derivative, slope, radius, radial_weights, extent):
    """Weak pressure-energy balances with global and compact radial tests."""
    inner_extent = extent / 2
    complement = (1 - (radius / inner_extent).square()).clamp_min(0)
    tests = torch.stack((torch.ones_like(radius), complement.pow(3)), dim=-1)
    slopes = torch.stack((
        torch.zeros_like(radius),
        -6 * radius / inner_extent**2 * complement.square(),
    ), dim=-1)
    return (
        radial_weights[..., None] * (
            tests * (pressure * (time_derivative + slope.square()))[..., None]
            + slopes * (pressure.square() * slope)[..., None]
        )
    ).sum(dim=1) / 1.6


def loss_fn(model, collocation, ic_points, bc_points, helpers):
    times, time_weights = _time_rule(collocation, helpers.T_MAX)
    # PDE sampling is detached; integral nodes below remain differentiable.
    with torch.no_grad():
        # The field and all physical partials vanish strictly inside its
        # analytic hole. Omit only those identically zero residual graphs;
        # keep the original point count in the mean below. This is the same
        # objective for every parameter state, with zero boundary contribution
        # when a site crosses the Darcy-compatible moving front.
        collocation_front, _, _ = model.front(collocation[:, 1:2])
        exterior = collocation[collocation[:, 0] >= collocation_front[:, 0]]
        front, _, _ = model.front(times[:, None])
        scale = front.clamp_max(0.6)
        unit_space = (
            torch.arange(8, device=collocation.device, dtype=collocation.dtype) + 0.5
        ) / 8
        distance = scale * unit_space / (
            1 - unit_space + scale / (helpers.R_MAX - front)
        )
        front_points = torch.stack((
            (front + distance).flatten(), times.repeat_interleave(len(unit_space)),
        ), dim=1)
    # Both point sets use the same physical PDE; one autodiff graph suffices.
    residual = _pde_residual(model, torch.cat((exterior, front_points)))
    count = len(exterior)
    physics = residual[:count].square().sum() / len(collocation)
    exterior_loss = (
        residual[count:].reshape(len(times), -1).square().mean(dim=1) * time_weights
    ).sum()

    # IC and Neumann losses are parameter independent for this ansatz. Keep
    # the same one-sided tangent constraint, using its analytic IC and RHS.
    advanced = ic_points.clone()
    advanced[:, 1] = helpers.dt
    tangent = (model(advanced) - helpers.initial(ic_points)) / helpers.dt
    tangent_loss = (tangent - _initial_rhs(ic_points[:, :1])).square().mean()

    front, _, _ = model.front(times[:, None])
    radius, weights = _exterior_rule(front, helpers.R_MAX)
    radial_weights = radius * weights
    points = torch.stack((
        radius.flatten(), times[:, None].expand_as(radius).flatten(),
    ), dim=1)
    values, physical_slope, physical_time = _pressure_jet(model, points, detach=False)
    pressure = values.reshape_as(radius)
    mass = (pressure * radial_weights).sum(dim=-1)
    mass_loss = ((mass - _initial_mass(helpers.R_MAX)).square() * time_weights).sum()
    # Differentiate w.r.t. physical coordinates first. Backpropagation of the
    # resulting integral then includes the parameter-dependent map and weights.
    time_derivative = physical_time.reshape_as(pressure)
    slope = physical_slope.reshape_as(pressure)
    moments = _moment_residuals(
        pressure, time_derivative, slope, radius, radial_weights, helpers.R_MAX,
    )
    energy = _energy_residuals(
        pressure, time_derivative, slope, radius, radial_weights, helpers.R_MAX,
    )
    flux = _flux_residuals(pressure, time_derivative, slope, radius, radial_weights)
    return (
        physics + tangent_loss + 0.25 * exterior_loss + mass_loss
        + 0.25 * (moments.square().mean(dim=1) * time_weights).sum()
        + 0.1 * (energy.square().mean(dim=1) * time_weights).sum()
        + 0.1 * (flux.square().mean(dim=1) * time_weights).sum()
    )
