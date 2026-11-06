import torch


def build_optimizer(model, config):
    return torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, max_eval=25,
                            history_size=50, line_search_fn="strong_wolfe",
                            tolerance_grad=1.0e-12, tolerance_change=1.0e-15)
