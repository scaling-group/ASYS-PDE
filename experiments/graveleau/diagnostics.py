"""Independent automatic-differentiation readouts of the frozen pressure field.

The radial mass in each row is density mass, pi*integral(r*v dr), since v=2*rho.
Adapted from the original saved-candidate assessment; grids are unchanged.
"""
import math
import torch

END = 0.995 * 0.6200901156700502


def derivative(value, points, dimension):
    return torch.autograd.grad(value.sum(), points, create_graph=True)[0][:, dimension:dimension + 1]


def measure(model):
    times = END * torch.tensor([0., .01, .025, .05, .1, .2, .35, .5, .65, .8, .9, .95, .98, 1.])
    # The pressure level is fixed independently of the official monitor.
    left, right = torch.full_like(times, .002), torch.full_like(times, 6.)
    with torch.no_grad():
        for _ in range(48):
            middle = (left + right) / 2
            positive = model(torch.stack((middle, times), dim=1)).flatten() > 1e-10
            right = torch.where(positive, middle, right)
            left = torch.where(positive, left, middle)
    rows = []
    for time, front in zip(times, right):
        radius = torch.linspace(.002, 6., 2049)
        points = torch.stack((radius, torch.full_like(radius, time)), dim=1).requires_grad_(True)
        value = model(points)
        vr, vt = derivative(value, points, 0), derivative(value, points, 1)
        diffusion = value * (derivative(vr, points, 0) + vr / points[:, :1])
        advection = vr.square()
        residual = vt - diffusion - advection
        active = value[:, 0] > 1e-8
        relative = (residual.square().sum() / (vt.square() + diffusion.square() + advection.square()).sum().clamp_min(1e-30)).sqrt()
        mass = math.pi * torch.trapezoid(value[:, 0] * radius, radius)
        offsets = min(float(front), .6) * torch.logspace(-5, .7, 257)
        near_radius = (front + offsets).clamp_max(6.)
        near = torch.stack((near_radius, torch.full_like(near_radius, time)), dim=1).requires_grad_(True)
        nv = model(near)
        nr, nt = derivative(nv, near, 0), derivative(nv, near, 1)
        nd = nv * (derivative(nr, near, 0) + nr / near[:, :1])
        ng = nr.square()
        nres = nt - nd - ng
        nrel = (nres.square().sum() / (nt.square() + nd.square() + ng.square()).sum().clamp_min(1e-30)).sqrt()
        rows.append({'t':float(time),'t_fraction':float(time / END),'front_pressure_level_1e-10':float(front),'mass':float(mass.detach()),'pressure_min':float(value.min().detach()),'uniform_residual_mse':float(residual.square().mean().detach()),'active_residual_mse':float(residual[active].square().mean().detach()),'relative_rms_equation_imbalance':float(relative.detach()),'front_log_offset_residual_mse':float(nres.square().mean().detach()),'front_relative_rms_equation_imbalance':float(nrel.detach())})
    with torch.no_grad():
        sample_r = torch.linspace(.002, 6., 2049)
        sample = torch.stack((sample_r, torch.zeros_like(sample_r)), dim=1).requires_grad_(True)
    initial = model(sample)
    initial_rhs = initial * (derivative(derivative(initial, sample, 0), sample, 0) + derivative(initial, sample, 0) / sample[:, :1]) + derivative(initial, sample, 0).square()
    initial_error = (initial - 1.6 * (1 - torch.exp(-((sample_r[:, None] - 1).clamp_min(0) / .6).square()))).abs().max()
    tangent_error = (derivative(initial, sample, 1) - initial_rhs).abs().max()
    masses = [r['mass'] for r in rows]
    return {'n_parameters':sum(p.numel() for p in model.parameters()),
            'raw_fitted_parameters':{k:v.detach().tolist() for k,v in model.state_dict().items()},
            'ic_max_absolute_error':float(initial_error.detach()),
            'true_initial_pde_tangent_max_absolute_error':float(tangent_error.detach()),
            'sampled_mass_range_fraction_of_initial':(max(masses)-min(masses))/masses[0],
            'rows':rows}
