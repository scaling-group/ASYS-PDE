"""Protected finite AD/stencil and pointwise/batch checks for this scalar profile."""
import torch
from task import field_derivatives


def check_field(model):
    points = torch.tensor([[.0013],[.0057],[.0091]], dtype=torch.float64)
    f, first, second, _ = field_derivatives(model, points)
    if not all(torch.isfinite(v).all() for v in (f, first, second)):
        raise ValueError('Nonfinite profile/derivatives')
    single = torch.cat([model(x[None,:]) for x in points])
    if not torch.allclose(f, single, atol=1e-12, rtol=1e-10):
        raise ValueError('Batch-coupled profile')
    errors = []
    for h in (3e-6, 1.5e-6):
        upper, lower = model(points+h), model(points-h)
        d1 = (upper-lower)/(2*h)
        d2 = (upper-2*f+lower)/h**2
        if not torch.isfinite(d1).all() or not torch.isfinite(d2).all():
            raise ValueError('Nonfinite stencil')
        if not torch.allclose(d1, first, atol=1e-8, rtol=2e-5):
            raise ValueError('First derivative mismatch')
        if not torch.allclose(d2, second, atol=3e-7, rtol=3e-5):
            raise ValueError('Second derivative mismatch')
        errors.append([float((d1-first).abs().max().detach()), float((d2-second).abs().max().detach())])
    parameters = [p for p in model.parameters() if p.requires_grad]
    grads = [g for i in range(len(points)) for g in torch.autograd.grad(f[i,0], parameters,
        retain_graph=True, allow_unused=True)] if parameters and f.requires_grad else []
    if not all(g is None or torch.isfinite(g).all() for g in grads):
        raise ValueError('Nonfinite parameter gradient')
    return {'passes': True, 'stencil_errors': errors,
        'parameter_gradient_nonzero_at_probes': any(g is not None and bool((g.abs()>1e-12).any()) for g in grads),
        'scope': 'Finite pointwise/batch/first/second derivative checks; no eighth-derivative or analytic regularity certificate'}
