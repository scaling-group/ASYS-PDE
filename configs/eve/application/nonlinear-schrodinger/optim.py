from asys_fit import _default_optimizer
import torch

def build_optimizer(model, config):
    optimizer = _default_optimizer(model, config)
    if isinstance(optimizer, torch.optim.LBFGS):
        optimizer.param_groups[0].update(tolerance_grad=1e-12, tolerance_change=1e-15)
    return optimizer
