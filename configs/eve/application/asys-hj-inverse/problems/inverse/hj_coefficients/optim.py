import torch


def make_optimizer(parameters, config):
    return torch.optim.Adam(parameters, lr=config["learning_rate"])
