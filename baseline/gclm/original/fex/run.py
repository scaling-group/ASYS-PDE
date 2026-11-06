"""Cold FEX-derived comparison runner; immutable attempts, no resume interface."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def choose(rows):
    """Return a public-selected identity, ignoring all reference/diagnostic fields."""
    valid = []
    for ordinal, row in enumerate(rows):
        public = row.get("public") or {}
        score = public.get("score")
        if (
            row["fit"]["completed"]
            and public.get("admissible")
            and isinstance(score, (int, float))
            and math.isfinite(score)
        ):
            valid.append((float(score), -ordinal, row["identity"]))
    return max(valid)[2] if valid else None


class Numerical:
    def __init__(self, packet, settings):
        if settings["mode"] not in ("target", "analytic-fixture"):
            raise ValueError("Unknown scientific mode")
        sys.path.insert(0, str(packet))
        sys.path.insert(0, str(packet / "runtime"))
        import torch
        from fit import check_profile, public_score
        from model import Model
        from operators import NormalizedProfile, Operators, losses
        from search import Controller, Pool, formula, optimize, reward

        self.torch, self.Model, self.Controller, self.Pool = torch, Model, Controller, Pool
        self.optimize, self.formula, self.reward = optimize, formula, reward
        self.public_score, self.check_profile = public_score, check_profile
        self.NormalizedProfile, self.losses = NormalizedProfile, losses
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        self.settings = settings
        self.operators = Operators(settings["training"]["quadrature"])
        count = settings["training"]["collocation"]
        self.batch = torch.exp((2 * (torch.arange(count) + 0.5) / count - 1) * math.log(1000))

    def controller(self, depth, seed):
        policy = self.Controller(
            depth, 700000 + depth * 100 + seed, self.settings["controller"]["temperature"]
        )
        optimizer = self.torch.optim.Adam(
            policy.parameters(), lr=self.settings["controller"]["learning_rate"]
        )
        return policy, optimizer

    def sample(self, policy, count):
        return policy.sample(count, self.settings["controller"]["exploration"]).tolist()

    def update(self, policy, optimizer, actions, rewards):
        optimizer.zero_grad()
        loss = policy.loss(
            self.torch.tensor(actions), rewards, self.settings["controller"]["retained_fraction"]
        )
        loss.backward()
        if not all(
            p.grad is None or bool(self.torch.isfinite(p.grad).all()) for p in policy.parameters()
        ):
            raise ValueError("Nonfinite controller gradient")
        optimizer.step()
        return float(loss.detach())

    def fit(self, path, config, stages, seconds, initial=None):
        path.mkdir(parents=True, exist_ok=False)
        (path / "model.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        model = self.Model(config)
        if initial is not None:
            model.load_state_dict(self.torch.load(initial / "state_dict.pt", weights_only=True))
        a = 0.25 if self.settings["mode"] == "target" else 0.0

        def objective(candidate):
            normalized = self.NormalizedProfile(candidate, self.operators)
            return (
                self.settings["training"]["physics_scale"]
                * self.losses(normalized, self.batch, self.operators, a=a)["physics"]
            )

        result = self.optimize(model, objective, stages, seconds)
        result["initialization"] = dict(
            kind="same_run_coarse" if initial else "fresh", source=initial.name if initial else None
        )
        if result["best_loss"] is not None:
            self.torch.save(model.state_dict(), path / "state_dict.pt")
            write(path / "formula.json", self.formula(model))
        write(path / "fit.json", result)
        return result

    def score(self, path):
        saved = path / "public.json"
        if saved.exists():
            return json.loads(saved.read_text())
        model = self.Model(yaml.safe_load((path / "model.yaml").read_text()))
        model.load_state_dict(self.torch.load(path / "state_dict.pt", weights_only=True))
        try:
            if self.settings["mode"] == "target":
                result = self.public_score(model)
            else:
                # Synthetic plumbing fixture only: analytic a=0 reference, no
                # target data. This scoring branch cannot run in target mode.
                profile, checks = self.check_profile(model)
                y = self.torch.logspace(-3, 3, 193)
                truth = y / (1 + y.square())
                error = float(
                    self.torch.linalg.vector_norm(profile(y) - truth)
                    / self.torch.linalg.vector_norm(truth)
                )
                result = dict(
                    admissible=math.isfinite(error),
                    score=-error,
                    analytic_fixture=True,
                    checks=checks,
                )
            if clean(result) != result:
                result.update(admissible=False, score=-1e9, failure="Nonfinite public diagnostic")
        except (ValueError, RuntimeError, OverflowError) as exc:
            result = dict(admissible=False, score=-1e9, failure=f"{type(exc).__name__}: {exc}")
        write(saved, result)
        return clean(result)

    def save_controller(self, policy, path):
        self.torch.save(
            dict(weights=policy.state_dict(), sampling_state=policy.generator.get_state()), path
        )


def run_case(packet, depth, seed, settings, engine, *, output=None):
    if depth not in settings["depths"] or seed not in settings["seeds"]:
        raise ValueError("Case outside the declared matrix")
    if settings["terminal"]["initialization"] != "retained_coarse_state_from_this_run":
        raise ValueError("Undeclared terminal initialization")
    root = Path(output) if output is not None else packet / "results" / f"depth-{depth}-seed-{seed}"
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    pool = engine.Pool(settings["pool_size"])
    policy, optimizer = engine.controller(depth, seed)
    fit_seconds = 0.0
    coarse_records, terminal_records, snapshots = [], [], []
    completed = False
    stop = "batch_limit"

    def remaining():
        return settings["case_wall_seconds"] - (time.monotonic() - started)

    def snapshot(label):
        rows = []
        for entry in pool.entries:
            identity = entry["identity"]
            path = root / "coarse" / identity
            rows.append(
                dict(
                    identity=identity,
                    fit=json.loads((path / "fit.json").read_text()),
                    public=engine.score(path),
                )
            )
        # Declaration order, not the pool's training-loss ordering, settles ties.
        rows.sort(key=lambda x: x["identity"])
        report = dict(
            label=label,
            proposals=len(coarse_records),
            fitting_seconds=fit_seconds,
            elapsed_seconds=time.monotonic() - started,
            rows=rows,
            selected=choose(rows),
        )
        snapshots.append(report)
        write(root / f"{label}.json", report)

    try:
        for batch_index in range(settings["controller"]["maximum_batches"]):
            actions = engine.sample(policy, settings["controller"]["batch_size"])
            attempted, rewards = [], []
            for action in actions:
                available = min(
                    settings["coarse"]["fit_seconds_per_proposal"],
                    settings["coarse"]["total_fit_seconds"] - fit_seconds,
                    settings["coarse"]["phase_wall_seconds"] - (time.monotonic() - started),
                    remaining(),
                )
                if available <= 0:
                    stop = "coarse_budget"
                    break
                ordinal = len(coarse_records)
                identity = f"proposal-{ordinal:05d}"
                config = dict(
                    depth=depth,
                    actions=action,
                    seed=1000000 + depth * 100000 + seed * 10000 + ordinal,
                    initial_beta=settings["initial_beta"],
                    initial_sd=settings["initial_sd"],
                )
                fit = engine.fit(
                    root / "coarse" / identity, config, settings["coarse"]["stages"], available
                )
                fit_seconds += fit["seconds"]
                effective_loss = fit["selection_loss"] if fit["completed"] else None
                pool.add(identity, action, effective_loss)
                attempted.append(action)
                rewards.append(engine.reward(effective_loss))
                row = dict(
                    identity=identity,
                    ordinal=ordinal,
                    actions=action,
                    fit=fit,
                    cumulative_fitting_seconds=fit_seconds,
                )
                coarse_records.append(row)
                with (root / "attempts.jsonl").open("a") as handle:
                    handle.write(json.dumps(clean(row), allow_nan=False) + "\n")
            policy_loss = (
                engine.update(policy, optimizer, attempted, rewards)
                if len(attempted) >= 2
                else None
            )
            with (root / "controller.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        dict(
                            batch=batch_index + 1,
                            attempted=len(attempted),
                            policy_loss=policy_loss,
                            rewards=rewards,
                        )
                    )
                    + "\n"
                )
            if batch_index + 1 in settings["coarse"]["snapshots_after_batches"]:
                snapshot(f"coarse-batch-{batch_index + 1}")
            if len(attempted) < len(actions):
                break
        snapshot("coarse-final")
        write(root / "frozen-pool.json", pool.entries)
        engine.save_controller(policy, root / "controller.pt")
        for ordinal, entry in enumerate(pool.entries):
            identity = f"terminal-{ordinal:02d}"
            origin = root / "coarse" / entry["identity"]
            config = yaml.safe_load((origin / "model.yaml").read_text())
            available = min(settings["terminal"]["fit_seconds_per_expression"], remaining())
            if available <= 0:
                fit = dict(
                    completed=False,
                    stop="not_attempted_case_deadline",
                    selection_loss=None,
                    seconds=0,
                )
                public = None
            else:
                path = root / "terminal" / identity
                fit = engine.fit(
                    path, config, settings["terminal"]["stages"], available, initial=origin
                )
                public = engine.score(path) if fit["completed"] else None
            terminal_records.append(
                dict(identity=identity, origin=entry["identity"], fit=fit, public=public)
            )
            write(root / "terminal-results.json", terminal_records)
        completed = True
    finally:
        result = dict(
            completed=completed,
            mode=settings["mode"],
            depth=depth,
            seed=seed,
            coarse_stop=stop,
            coarse_attempts=len(coarse_records),
            coarse_failures=sum(not x["fit"]["completed"] for x in coarse_records),
            coarse_censored=sum(x["fit"]["stop"] == "wall_time" for x in coarse_records),
            coarse_fitting_seconds=fit_seconds,
            terminal=terminal_records,
            terminal_fitting_seconds=sum(x["fit"]["seconds"] for x in terminal_records),
            public_selected=choose(terminal_records),
            snapshots=[{k: v for k, v in r.items() if k != "rows"} for r in snapshots],
            elapsed_seconds=time.monotonic() - started,
        )
        write(root / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = Path(__file__).resolve().parent
    settings = yaml.safe_load((packet / "study.yaml").read_text())
    backend = Numerical(packet, settings)
    result = run_case(packet, args.depth, args.seed, settings, backend, output=args.output)
    print(json.dumps({k: result[k] for k in ("completed", "coarse_attempts", "public_selected", "elapsed_seconds")}))
