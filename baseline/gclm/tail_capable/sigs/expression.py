"""Explicit literal-to-parameter interpretation for a SIGS adapter.

No expression repair, clipping, protected division, or target-specific atoms.
Integer exponents and mathematical constants stay fixed, following SIGS
Appendix C. All other numeric occurrences are independent scalar parameters.
"""

import ast
import math

import torch

FUNCTIONS = {name: getattr(torch, name) for name in ("sin", "cos", "exp", "log", "sqrt", "tanh")}


def number(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = number(node.operand)
        if value is not None:
            return -value if isinstance(node.op, ast.USub) else value
    return None


class Expression(torch.nn.Module):
    def __init__(self, text):
        super().__init__()
        self.original = text
        tree = ast.parse(text.replace("^", "**"), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 512:
            raise ValueError("Expression exceeds the declared syntax limit")
        initial = []

        def convert(node, protect=False):
            value = number(node)
            if value is not None:
                if not math.isfinite(value):
                    raise ValueError("Nonfinite literal")
                if protect:
                    return ("constant", value), ast.Constant(value)
                index = len(initial)
                initial.append(value)
                return ("parameter", index), ast.Name(id=f"p{index}", ctx=ast.Load())
            if isinstance(node, ast.Name) and node.id in ("x", "pi", "e"):
                return ("name", node.id), node
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                child, symbolic = convert(node.operand)
                sign = -1 if isinstance(node.op, ast.USub) else 1
                return ("sign", sign, child), ast.UnaryOp(op=node.op, operand=symbolic)
            if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
            ):
                exponent = number(node.right) if isinstance(node.op, ast.Pow) else None
                fixed = exponent is not None and math.isfinite(exponent) and exponent.is_integer()
                left, sl = convert(node.left)
                right, sr = convert(node.right, protect=fixed)
                return (type(node.op).__name__, left, right), ast.BinOp(
                    left=sl, op=node.op, right=sr
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in FUNCTIONS
                and len(node.args) == 1
                and not node.keywords
            ):
                child, symbolic = convert(node.args[0])
                return ("function", node.func.id, child), ast.Call(
                    func=node.func, args=[symbolic], keywords=[]
                )
            raise ValueError(f"Unsupported syntax: {ast.dump(node)}")

        self.tree, symbolic = convert(tree.body)
        self.parameterized = ast.unparse(ast.fix_missing_locations(symbolic))
        self.coefficients = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float64))

    def forward(self, x):
        def evaluate(node):
            kind = node[0]
            if kind == "constant":
                return x * 0 + node[1]
            if kind == "parameter":
                return x * 0 + self.coefficients[node[1]]
            if kind == "name":
                return x if node[1] == "x" else x * 0 + getattr(math, node[1])
            if kind == "sign":
                return node[1] * evaluate(node[2])
            if kind == "function":
                return FUNCTIONS[node[1]](evaluate(node[2]))
            left, right = evaluate(node[1]), evaluate(node[2])
            if kind == "Add":
                return left + right
            if kind == "Sub":
                return left - right
            if kind == "Mult":
                return left * right
            if kind == "Div":
                return left / right
            return left**right

        return evaluate(self.tree)


class Profile(torch.nn.Module):
    """Supplied envelope with an expression in s=(1+x²)^(-1/(2β)).

    Admissibility of a proposed expression over the whole positive s interval
    is a separate search requirement; this wrapper does not manufacture it.
    """

    def __init__(self, expression, beta=1.0):
        super().__init__()
        self.expression = Expression(expression)
        self.log_beta = torch.nn.Parameter(torch.tensor(math.log(beta), dtype=torch.float64))

    def beta(self):
        return self.log_beta.exp()

    def forward(self, x):
        s = torch.exp(-0.5 / self.beta() * torch.log1p(x.square()))
        correction = self.expression(s) - self.expression(torch.ones_like(s))
        return x * torch.exp(-(1 + 1 / self.beta()) / 2 * torch.log1p(x.square()) + correction)
