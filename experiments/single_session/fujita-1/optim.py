"""Cold coefficient-fitting recipe, bounded by the evaluator deadline."""
import torch


def build_stepper(params, helpers):
    return torch.optim.LBFGS(params, lr=0.5, max_iter=10, max_eval=15,
                             history_size=40,
                             tolerance_grad=1e-9, tolerance_change=1e-12,
                             line_search_fn="strong_wolfe")
