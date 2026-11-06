"""Relative weak correction residual with fixed analytical training tensors."""
import math
import numpy as np
import torch
from problem import TRAIN_WIDTHS, T_MAX


def time_evaluation(model, times):
    u, z = times/T_MAX, 2*times/T_MAX-1
    values = [torch.ones_like(z), z]
    derivatives = [torch.zeros_like(z), torch.full_like(z, 2/T_MAX)]
    for _ in range(2, model.time_degree+1):
        derivatives.append(4/T_MAX*values[-1]+2*z*derivatives[-1]-derivatives[-2])
        values.append(2*z*values[-1]-values[-2])
    value, derivative = torch.stack(values, dim=-1), torch.stack(derivatives, dim=-1)
    return u[:, None].square()*value, 2*u[:, None]/T_MAX*value+u[:, None].square()*derivative


def make_cache(model, config):
    from model import line_field
    dtype, device = model.coefficients.dtype, model.coefficients.device
    nx, nt = int(config.get('training_space_nodes', 256)), int(config.get('training_time_nodes', 320))
    nodes, weights = np.polynomial.legendre.leggauss(nt)
    times = torch.as_tensor((nodes+1)*T_MAX/2, dtype=dtype, device=device)
    weights = torch.as_tensor(weights/2, dtype=dtype, device=device)
    widths = torch.as_tensor(TRAIN_WIDTHS, dtype=dtype, device=device)
    value, derivative = time_evaluation(model, times)
    polynomials = torch.stack([model.basis(w) for w in widths])
    kernels = torch.einsum('tj,wjnk->wtnk', value.to(polynomials.dtype), polynomials)
    dt = torch.einsum('tj,wjnk->wtnk', derivative.to(polynomials.dtype), polynomials)
    omega = 0.5*(math.pi*model.feedback_n[None, :]/widths[:, None]).square()
    linear = 1j*dt-omega[:, None, :, None]*kernels
    x = ((2*torch.arange(nx, dtype=dtype, device=device)+1)/nx-1)[None, :]*widths[:, None]
    xx = x[:, None, :].expand(-1, nt, -1)
    tt, ll = times[None, :, None].expand_as(xx), widths[:, None, None].expand_as(xx)
    c = line_field(xx.reshape(-1, 1), tt.reshape(-1, 1), ll.reshape(-1, 1)).reshape_as(xx)
    cosr = torch.cos(math.pi*x[:, :, None]*model.radiation_n/widths[:, None, None])
    cosine = torch.cos(math.pi*x[:, :, None]*model.feedback_n/widths[:, None, None])
    mr, mf = torch.where(model.radiation_n == 0, 1.0, 2.0), torch.where(model.feedback_n == 0, 1.0, 2.0)
    b = torch.einsum('wtn,wxn->wtx', model.radiation_amplitudes(
        times[None, :].expand(len(widths), -1), widths[:, None].expand(-1, nt)), (cosr*mr).to(c.dtype))
    residual_weights = (weights[None, :, None]*mf[None, None, :]/(len(widths)*mf.sum())).expand(len(widths), -1, -1)
    predictor = kernels[..., 0]*model.seed_norms[0]*model.recurrence[0, 0]
    predictor = torch.einsum('wtn,wxn->wtx', predictor, (cosine*mf).to(c.dtype))
    return dict(core=c, radiation=b, fixed=c+b, background=c+b+predictor,
                analysis=cosine.to(c.dtype)/nx, reconstruction=(cosine*mf).to(c.dtype),
                kernels=kernels, linear=linear, scale=torch.exp(-widths)/widths,
                residual_weights=residual_weights, widths=widths, times=times)


def residual_basis(model, cache):
    field = torch.einsum('wtnk,wxn->wtxk', cache['kernels'], cache['reconstruction'])
    c = cache['background'][..., None]
    nonlinear = torch.einsum('wtxk,wxn->wtnk',
        2*c.abs().square()*field+c.square()*field.conj(), cache['analysis'])
    return (cache['linear']+nonlinear)/cache['scale'][:, None, None, None]


def loss_fn(model, collocation, ic_points, bc_points, helpers):
    from model import cubic_difference
    if model._training_cache is None:
        with torch.no_grad():
            model._training_cache = make_cache(model, helpers.config)
    cache = model._training_cache
    total = model.total_coefficients()
    shared = (model.preconditioner@total).to(cache['kernels'].dtype)
    d = cache['kernels']@shared
    correction = cache['radiation']+torch.einsum('wtn,wxn->wtx', d, cache['reconstruction'])
    nonlinear = torch.einsum('wtx,wxn->wtn', cubic_difference(cache['core'], correction), cache['analysis'])
    residual = (cache['linear']@shared+nonlinear)/cache['scale'][:, None, None]
    return (residual.abs().square()*cache['residual_weights']).sum()+float(
        helpers.config.get('coefficient_ridge', 1e-10))*total.square().mean()
