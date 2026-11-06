import torch

def build_optimizer(model, cfg):
    return torch.optim.LBFGS(model.parameters(), lr=float(cfg.get('lr', .2)),
        max_iter=1, history_size=20, line_search_fn='strong_wolfe')
