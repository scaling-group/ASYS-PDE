"""Finite expression tree inside the supplied gCLM envelope.

Its grammar contains only the original FEX elementary operators.
"""

import math

import torch

UNARY = ("zero", "one", "identity", "square", "cube", "fourth", "exp", "sin", "cos")
BINARY = ("add", "multiply", "subtract")


def topology(depth):
    """Preorder slots for binary trees with unary children; depth one has 3 slots."""
    if depth not in (1, 2, 3):
        raise ValueError("Declared tree depths are one, two and three")
    nodes = []
    count = 0

    def unary(remaining):
        nonlocal count
        index = len(nodes)
        parameter = count
        count += 1
        nodes.append({"kind": "unary", "parameter": parameter})
        nodes[index]["child"] = branch(remaining) if remaining else None
        return index

    def branch(remaining):
        index = len(nodes)
        nodes.append({"kind": "binary"})
        nodes[index]["left"] = unary(remaining - 1)
        nodes[index]["right"] = unary(remaining - 1)
        return index

    branch(depth)
    return nodes


def primitive(action, x):
    if action == 0:
        return x * 0
    if action == 1:
        return x * 0 + 1
    if action == 2:
        return x
    if action in (3, 4, 5):
        return x ** (action - 1)
    if action == 6:
        return torch.exp(x)
    if action == 7:
        return torch.sin(x)
    if action == 8:
        return torch.cos(x)
    raise ValueError("Unknown unary operation")


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nodes = topology(int(config["depth"]))
        self.actions = tuple(int(v) for v in config["actions"])
        if len(self.nodes) != len(self.actions):
            raise ValueError("Action count differs from tree size")
        for node, action in zip(self.nodes, self.actions, strict=True):
            choices = len(UNARY) if node["kind"] == "unary" else len(BINARY)
            if not 0 <= action < choices:
                raise ValueError("Action outside its node's operator set")
        size = sum(node["kind"] == "unary" for node in self.nodes)
        generator = torch.Generator().manual_seed(int(config.get("seed", 0)))
        initial = torch.randn(size, 2, generator=generator, dtype=torch.float64)
        self.coefficients = torch.nn.Parameter(initial * float(config.get("initial_sd", 0.1)))
        self.log_beta = torch.nn.Parameter(
            torch.tensor(math.log(float(config.get("initial_beta", 1))), dtype=torch.float64)
        )

    def beta(self):
        return torch.exp(self.log_beta)

    def expression(self, z, index=0):
        node, action = self.nodes[index], self.actions[index]
        if node["kind"] == "unary":
            child = z if node["child"] is None else self.expression(z, node["child"])
            coefficient = self.coefficients[node["parameter"]]
            return coefficient[0] * primitive(action, child) + coefficient[1]
        left = self.expression(z, node["left"])
        right = self.expression(z, node["right"])
        if action == 0:
            return left + right
        if action == 1:
            return left * right
        return left - right

    def forward(self, y):
        z = 1 - 2 * torch.exp(-0.5 / self.beta() * torch.log1p(y.square()))
        correction = self.expression(z) - self.expression(torch.full_like(z, -1))
        power = (1 + 1 / self.beta()) / 2
        return y * torch.exp(-power * torch.log1p(y.square()) + correction)
