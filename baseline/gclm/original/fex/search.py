"""FEX-derived controller, pool and coefficient fitting; no target run entrypoint.

Uses the JMLR paper's reward and all-candidate pool, with declared finite-budget
and gCLM adaptations. Failure records and original states are never overwritten.
"""

import copy
import math
import time

import torch
from model import BINARY, UNARY, topology


class Controller(torch.nn.Module):
    def __init__(self, depth, seed, temperature=5.0):
        super().__init__()
        self.choices = [
            len(UNARY) if n["kind"] == "unary" else len(BINARY) for n in topology(depth)
        ]
        self.temperature = temperature
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = torch.nn.Sequential(
                torch.nn.Linear(20, 50), torch.nn.ReLU(), torch.nn.Linear(50, sum(self.choices))
            ).double()
        self.generator = torch.Generator().manual_seed(seed + 1)

    def logits(self):
        result = self.network(torch.zeros(1, 20, dtype=torch.float64))[0] / self.temperature
        return result.split(self.choices)

    def sample(self, count, exploration=0.1):
        if count < 1 or not 0 <= exploration <= 1:
            raise ValueError("Invalid sampling request")
        columns = []
        for logits in self.logits():
            probabilities = (1 - exploration) * logits.softmax(0) + exploration / len(logits)
            columns.append(
                torch.multinomial(probabilities, count, replacement=True, generator=self.generator)
            )
        return torch.stack(columns, dim=1)

    def log_probabilities(self, actions):
        # The paper's Eq. 17 uses base-policy log probabilities with epsilon
        # exploration.
        return torch.stack(
            [logits.log_softmax(0)[actions[:, i]] for i, logits in enumerate(self.logits())], dim=1
        ).sum(1)

    def loss(self, actions, rewards, retained_fraction=0.5):
        if not 0 < retained_fraction <= 1:
            raise ValueError("Invalid retained fraction")
        rewards = torch.as_tensor(rewards, dtype=torch.float64)
        if len(rewards) != len(actions) or not bool(torch.isfinite(rewards).all()):
            raise ValueError("Expected one finite reward per action sequence")
        threshold = torch.quantile(rewards, 1 - retained_fraction, interpolation="linear")
        advantage = (rewards - threshold).clamp_min(0).detach()
        return -(advantage * self.log_probabilities(actions)).mean()


def reward(loss):
    return 1 / (1 + loss) if loss is not None and math.isfinite(loss) and loss >= 0 else 0.0


class Pool:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("Pool must be nonempty")
        self.capacity = capacity
        self.entries = []

    def add(self, identity, actions, loss):
        if reward(loss) == 0:
            return
        actions = tuple(actions)
        previous = next((x for x in self.entries if x["actions"] == actions), None)
        if previous and previous["loss"] <= loss:
            return
        if previous:
            self.entries.remove(previous)
        self.entries.append(dict(identity=identity, actions=actions, loss=float(loss)))
        # Stable order preserves first occurrence for ties; no field-error input.
        self.entries.sort(key=lambda x: x["loss"])
        del self.entries[self.capacity :]


def optimize(model, objective, stages, wall_seconds):
    """Run one predeclared fit; bounded stages share a state, never an old run.

    The objective receives the current model. Return the best evaluated state,
    not an unmeasured last optimizer step. Numerical failure rejects the fit.
    """
    if wall_seconds <= 0:
        raise ValueError("Positive fit ceiling required")
    started = time.monotonic()
    best_loss, best_state, evaluations = math.inf, None, 0
    trace = []
    stage_counts = []

    class FitDeadline(Exception):
        pass

    def closure():
        nonlocal best_loss, best_state, evaluations
        if time.monotonic() - started >= wall_seconds:
            raise FitDeadline
        model.zero_grad(set_to_none=True)
        value = objective(model)
        if value.ndim or not bool(torch.isfinite(value)):
            raise ValueError("Nonfinite or nonscalar objective")
        value.backward()
        if any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        ):
            raise ValueError("Nonfinite coefficient gradient")
        loss = float(value.detach())
        evaluations += 1
        if loss < best_loss:
            best_loss, best_state = loss, copy.deepcopy(model.state_dict())
        return value

    stop = "stage_limits"
    failure = None
    try:
        closure()
        for spec in stages:
            if spec["kind"] == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=spec["lr"])
            elif spec["kind"] == "lbfgs":
                optimizer = torch.optim.LBFGS(
                    model.parameters(),
                    lr=spec["lr"],
                    max_iter=1,
                    max_eval=20,
                    history_size=50,
                    tolerance_grad=1e-10,
                    tolerance_change=1e-14,
                    line_search_fn="strong_wolfe",
                )
            else:
                raise ValueError("Unknown optimizer stage")
            counts = dict(kind=spec["kind"], completed_updates=0)
            stage_counts.append(counts)
            for update in range(spec["updates"]):
                if spec.get("cosine", False):
                    rate = spec["lr"] * 0.5 * (1 + math.cos(math.pi * update / spec["updates"]))
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                if spec["kind"] == "adam":
                    closure()
                    optimizer.step()
                else:
                    optimizer.step(closure)
                counts["completed_updates"] += 1
            closure()  # Observe the last update before changing stages.
            trace.append(dict(**counts, best_loss=best_loss, seconds=time.monotonic() - started))
    except FitDeadline:
        stop = "wall_time"
    except (ValueError, RuntimeError, OverflowError) as exc:
        stop, failure = "failed", f"{type(exc).__name__}: {exc}"
    if best_state is not None:
        model.load_state_dict(best_state)
    return dict(
        completed=failure is None and best_state is not None,
        stop=stop,
        failure=failure,
        best_loss=best_loss if best_state is not None else None,
        selection_loss=best_loss if failure is None and best_state is not None else None,
        evaluations=evaluations,
        stages=stage_counts,
        trace=trace,
        seconds=time.monotonic() - started,
    )


def formula(model):
    """Readable tree with round-trip coefficients, outside the candidate program."""
    coefficients = model.coefficients.detach().tolist()

    def expression(index):
        node, action = model.nodes[index], model.actions[index]
        if node["kind"] == "binary":
            symbol = ("+", "*", "-")[action]
            return f"({expression(node['left'])} {symbol} {expression(node['right'])})"
        value = "z" if node["child"] is None else expression(node["child"])
        name = UNARY[action]
        term = {"zero": "0", "one": "1", "identity": value}.get(name)
        if term is None:
            term = f"({value})**{action - 1}" if action in (3, 4, 5) else f"{name}({value})"
        a, b = coefficients[node["parameter"]]
        return f"({a!r}*({term}) + {b!r})"

    return dict(
        variable="z=2*x**2/(1+x**2)-1",
        expression=expression(0),
        raw_profile="h(x)=x*exp(-(1+1/beta)/2*log1p(x**2)+E(z)-E(-1))",
        beta=float(model.beta().detach()),
        coefficients=coefficients,
        fitted_parameters=sum(p.numel() for p in model.parameters()),
        tree_nodes=len(model.nodes),
        buffers=sum(p.numel() for p in model.buffers()),
        note=(
            "All coefficients counted, including structurally redundant constants; "
            "no simplification or refit."
        ),
    )
