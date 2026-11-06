"""Editable coefficient-fitting recipe, bounded by the evaluator deadline."""

import torch


def build_stepper(params, helpers):
    return torch.optim.LBFGS(
        params,
        lr=0.8,
        max_iter=1,
        max_eval=25,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
