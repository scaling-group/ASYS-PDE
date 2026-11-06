import torch

def build_optimizer(model, cfg):
    return torch.optim.LBFGS(model.parameters(), lr=float(cfg.get('lr', 1.)),
        max_iter=1, history_size=40, tolerance_grad=1e-16,
        tolerance_change=1e-22, line_search_fn='strong_wolfe')
