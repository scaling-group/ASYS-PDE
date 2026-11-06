import math
import torch


class AnnealedAdam(torch.optim.Adam):
    """Cosine decay reduces minibatch noise in the final spectral amplitudes."""

    def __init__(self, parameters, config):
        super().__init__(parameters, lr=config["learning_rate"])
        self.initial_lr = config["learning_rate"]
        self.total_steps = config["steps"]
        self.updates = 0

    def step(self, closure=None):
        fraction = min(self.updates / max(1, self.total_steps - 1), 1.0)
        rate = self.initial_lr * (0.02 + 0.49 * (1 + math.cos(math.pi * fraction)))
        for group in self.param_groups:
            group["lr"] = rate
        result = super().step(closure)
        self.updates += 1
        return result


def make_optimizer(parameters, config):
    return AnnealedAdam(parameters, config)
