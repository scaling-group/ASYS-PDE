"""Source rules shared by official evaluation and local workspace checks."""

import ast


def validate_sources(case):
    for name in ("model.py", "loss.py", "optim.py"):
        tree = ast.parse((case / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for module in modules:
                    if module.split(".")[0] not in {
                        "math",
                        "torch",
                        "problems",
                        "__future__",
                    }:
                        raise ValueError(
                            f"{name}: candidate import is not a mathematical dependency: {module}"
                        )
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in {
                    "open",
                    "eval",
                    "exec",
                    "__import__",
                    "compile",
                }:
                    raise ValueError(
                        f"{name}: candidate file/code access is prohibited"
                    )
                if isinstance(func, ast.Attribute) and func.attr in {
                    "load",
                    "save",
                    "loadtxt",
                    "from_file",
                    "solve_ivp",
                    "odeint",
                }:
                    raise ValueError(
                        f"{name}: external state or numerical solution construction is prohibited"
                    )
