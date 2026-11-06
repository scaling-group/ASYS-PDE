"""Equation-only fit; the ansatz fixes lower traces and the logarithmic resonance.

No numerical reference or scorer grids are used in training.
"""
def loss_fn(model, batch, task):
    bulk = task.normalized_residual(model, batch['bulk'])
    early = task.normalized_residual(model, batch['early'])
    return bulk.square().mean() + early.square().mean()
