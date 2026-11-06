import torch


def build_stepper(params, helpers):
    return torch.optim.Adam(params, lr=helpers["config"].get("learning_rate", 0.01))
