import torch


class Candidate(torch.nn.Module):
    """Declared unfitted even algebraic envelope, replaceable by the worker."""

    def __init__(self):
        super().__init__()
        self.amplitude = torch.nn.Parameter(torch.tensor(1.0))
        self.log_width = torch.nn.Parameter(torch.tensor(-4.605170185988091))

    def forward(self, eta):
        return self.amplitude / torch.sqrt(1 + self.log_width.exp() * eta.square())
