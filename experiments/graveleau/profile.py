"""The manuscript's normalized profile components, with user-supplied shape coefficients."""
import math

import torch


def polynomial(x, coefficients):
    value = torch.zeros_like(x)
    for coefficient in reversed(coefficients):
        value = value*x+coefficient
    return value


def profile(eta, beta, coefficients, repaired=True):
    if not .5 < beta < 1 or bool((eta < 1).any()):
        raise ValueError('Require 0 < p=2-1/beta < 1 and eta >= 1')
    p = 2-1/beta
    log_eta = eta.log()
    x = -torch.expm1((p-2)*log_eta) if repaired else -torch.expm1(-log_eta)
    envelope = torch.exp(x*polynomial(x,coefficients))
    return beta*torch.exp(p*log_eta)*x*envelope if repaired else torch.expm1(p*log_eta)/p*envelope


def tail(beta, coefficients):
    if not .5 < beta < 1:
        raise ValueError('Require 0 < p=2-1/beta < 1')
    p = 2-1/beta
    p1 = sum(coefficients)
    dp1 = sum(j*c for j,c in enumerate(coefficients))
    c = beta*math.exp(p1)
    d = -c*(1+p1+dp1)
    return dict(p=p,C=c,D=d,required_D=-2*c*c*p*p/(2-p),
                earlier_constant_offset=-math.exp(p1)/p,
                earlier_limiting_equation_defect=-math.exp(p1))


def fitted_coefficients(model, time):
    """Return beta and ascending coefficients of P at a physical time.

    The original front_shape stores bounded tensor-product shifted Legendre
    coefficients, degree two in normalized time and in the fractional space
    coordinate. This conversion includes the model's analytic edge offset.
    """
    theta = time / (0.995 * 0.6200901156700502)
    temporal = model.front_shape.new_tensor((1., 2*theta-1, 6*theta**2-6*theta+1))
    modes = temporal @ (3*torch.tanh(model.front_shape/3))
    beta = .6 + .38*torch.sigmoid(model.exponent)
    constant = modes[0]-modes[1]+modes[2] + 1.75-2*beta
    coefficients = torch.stack((constant, 2*modes[1]-6*modes[2], 6*modes[2]))
    return float(beta.detach()), coefficients.detach().tolist()
