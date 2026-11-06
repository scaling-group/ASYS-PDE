"""Protected conditional profile equation and finite scoring contract.

This is a local mathematical component, not a radial initial-value solution.
The state fixes normalization C=1; no numerical reference is used.
"""
import importlib.util
import sys
from pathlib import Path
import torch
import yaml

GAMMA = 7/3
XMAX = .01


def load_model(path):
    spec = importlib.util.spec_from_file_location('profile_candidate', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.Candidate().double()
    cfg = yaml.safe_load(Path(path).with_name('config.yaml').read_text())
    if 'initial_state' in cfg:
        existing = model.state_dict()
        compatible = {}
        for name, values in cfg['initial_state'].items():
            if name in existing:
                value = torch.tensor(values, dtype=existing[name].dtype)
                if value.shape == existing[name].shape:
                    compatible[name] = value
        model.load_state_dict(compatible, strict=False)
    return model


def gradient(value, points):
    if not value.requires_grad:
        return torch.zeros_like(points)
    answer = torch.autograd.grad(value.sum(), points, create_graph=True,
        retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(points) if answer is None else answer


def field_derivatives(model, points):
    x = points.clone().requires_grad_(True)
    f = model(x)
    if f.shape != (len(x), 1):
        raise ValueError('One profile value per coordinate required')
    first = gradient(f, x)
    second = gradient(first, x)
    residual = f*second + first.square() - GAMMA*x*first + (2*GAMMA-1)*f
    return f, first, second, residual


def traces(model):
    x = torch.zeros((1,1), requires_grad=True)
    f = model(x)
    first = gradient(f, x)
    second = gradient(first, x)
    third = gradient(second, x)
    return f, first, second, third


def samples(seed, multiplier=1):
    u = torch.quasirandom.SobolEngine(1, scramble=True, seed=seed).draw(256*multiplier).double()
    return {'bulk': .001 + .009*u, 'early': torch.exp(torch.log(u.new_tensor(1e-4))
        + u*torch.log(u.new_tensor(10.)))}


def normalized_residual(model, points):
    x = points.clone().requires_grad_(True)
    f = model(x)
    first = gradient(f, x)
    second = gradient(first, x)
    terms = (f*second, first.square(), -GAMMA*x*first, (2*GAMMA-1)*f)
    scale = sum(t.abs() for t in terms)
    return sum(terms)/torch.where(scale>0, scale, torch.ones_like(scale))


def terms(model, batch):
    f, first, second, third = traces(model)
    return {'bulk_physics': normalized_residual(model, batch['bulk']).square().mean(),
        'early_physics': normalized_residual(model, batch['early']).square().mean(),
        'weak': f.new_zeros(()),
        'initial': (f.square()+first.square()).mean(),
        'tangent': ((second-1/3)/(1/3)).square().mean(),
        'boundary': ((third-6)/6).square().mean()}


def physical_checks(model):
    x = torch.cat((torch.zeros((1,1)), torch.linspace(1e-5, XMAX, 257)[:,None]))
    f, first, second, residual = field_derivatives(model, x)
    jets = traces(model)
    finite = all(torch.isfinite(v).all() for v in (f, first, second, residual, *jets))
    targets = (0., 0., 1/3, 6.)
    errors = [float((v.detach()-target).abs().max()) for v,target in zip(jets, targets)]
    positive = bool((f[1:] > 0).all() and (first >= -1e-10).all()) if finite else False
    normalized = errors[0] <= 1e-12 and errors[1] <= 1e-12 and errors[2] <= 1e-7 and errors[3] <= 1e-6
    return {'passes': bool(finite and positive and normalized), 'finite': bool(finite),
        'positive_monotone_at_probes': positive, 'origin_jet_errors': errors,
        'scope': 'Finite profile positivity/monotonicity and supplied C=1 normalization; not existence, whole-line, waiting persistence or radial IVP certification'}
