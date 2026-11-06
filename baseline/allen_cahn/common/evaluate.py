"""Native AC task evaluation: fit candidate parameters, then score the fixed equation."""
import sys

# Direct worker calls must not create protected candidate bytecode files.
sys.dont_write_bytecode = True

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

import torch
import yaml

FAILURE = {key: -1.0e9 for key in ("score", "physics", "ic", "ic_tangent", "bc")}
SOURCE_FILES = ("model.py", "loss.py", "optim.py", "config.yaml", "problem.py", "evaluate.py")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def load(workspace, overrides=None):
    # Four scientific source files plus a native fitted-state artifact folder.
    sys.path.insert(0, str(workspace))
    problem = module("problem", workspace / "problem.py")
    config = yaml.safe_load((workspace / "config.yaml").read_text())
    config.update(overrides or {})
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    model = module("model", workspace / "model.py").Candidate(copy.deepcopy(config)).double().cpu()
    return problem, config, model


def public_score(problem, model):
    generator = torch.Generator().manual_seed(2299)
    p, _, b = problem.sample((4096, 1, 128), generator)
    axis = torch.arange(32, dtype=torch.float64) / 16.0 - 1.0
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    i = torch.stack((x.flatten(), y.flatten(), torch.zeros(1024)), dim=1)
    model.eval()
    # The prescribed trace is not smoothly periodic at the seam: its local
    # initial PDE derivative is scored strictly inside the spatial domain.
    j = i[(i[:, 0].abs() < 1.0) & (i[:, 1].abs() < 1.0)]
    metrics = problem.terms(model, p, i, b, jet_points=j)
    values = {key: -float(metrics[key].detach())
              for key in ("physics", "ic", "ic_tangent", "bc")}
    if not all(torch.isfinite(torch.tensor(v)) for v in values.values()):
        raise FloatingPointError("Nonfinite public equation score")
    values["score"] = sum(values.values()) / 4.0
    values["summary"] = "fixed strong residual, initial trace, exact interior initial derivative and periodicity"
    return values


def save_artifacts(model, config, log, history):
    torch.save(model.state_dict(), log / "state.pt")
    (log / "effective_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    torch.save(history, log / "training_history.pt")
    sampled = history[::max(1, (len(history) + 31) // 32)]
    if history and (not sampled or sampled[-1] != history[-1]):
        sampled.append(history[-1])
    (log / "training.json").write_text(json.dumps(
        {"closure_count": len(history), "sampled_closures": sampled}, indent=2) + "\n")


def frozen_public_score(log):
    """Score the persisted field, excluding fitting-time objects and globals."""
    scoring_log = log / "frozen_score"
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    command = [sys.executable, "-B", str(log / "source" / "evaluate.py"),
               "--workspace", str(log / "source"), "--log", str(scoring_log),
               "--score-only", "--start-state", str(log / "state.pt"),
               "--effective-config", str(log / "effective_config.yaml")]
    with (log / "scoring_stdout.txt").open("w") as out, (log / "scoring_stderr.txt").open("w") as err:
        subprocess.run(command, check=True, stdout=out, stderr=err)
    if (scoring_log / "error.txt").exists():
        raise RuntimeError("Frozen field scoring failed; see frozen_score/error.txt")
    receipt = json.loads((scoring_log / "manifest.json").read_text())
    assert receipt["optimizer_steps_completed"] == receipt["closure_calls"] == 0
    assert receipt["process_id"] != os.getpid()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return yaml.safe_load((scoring_log / "score.yaml").read_text()), {
        "public_scoring_contract": "fresh interpreter, frozen source/config/state",
        "scoring_process_id": receipt["process_id"],
        "scoring_seconds": time.monotonic() - started,
        "scoring_cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        "scoring_optimizer_steps": 0,
    }


def fit(problem, config, model, workspace, log, seconds):
    private = problem.sample((config["n_collocation"], config["n_ic"], config["n_bc"]),
                             torch.Generator().manual_seed(int(config["seed"]) + 1))
    loss_fn = module("loss", workspace / "loss.py").loss_fn
    trainable = [p for p in model.parameters() if p.requires_grad]
    initial_state = copy.deepcopy(model.state_dict())
    best_state = copy.deepcopy(initial_state)
    best_loss = float("inf")
    start = time.monotonic()
    cpu_start = time.process_time()
    history = []
    step = 0
    completed_steps = 0
    closures = 0
    best_at = start
    reason = "no trainable parameters"
    model.train()
    if trainable:
        optimizer = module("optim", workspace / "optim.py").build_optimizer(model, config)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("build_optimizer must return torch.optim.Optimizer")

        def closure():
            nonlocal best_loss, best_state, best_at, closures
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, *private, problem)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise FloatingPointError("Training loss must be one finite scalar")
            loss.backward()
            closures += 1
            number = float(loss.detach())
            history.append({"step": step, "closure": closures,
                            "seconds": time.monotonic() - start, "loss": number})
            if number < best_loss:
                best_loss = number
                best_state = copy.deepcopy(model.state_dict())
                best_at = time.monotonic()
            return loss

        reason = "max_steps"
        # Reserve scoring time. A single optimizer step cannot be forcibly interrupted.
        deadline = start + max(0.0, seconds - min(30.0, seconds * 0.1))
        for step in range(1, int(config["max_steps"]) + 1):
            if time.monotonic() >= deadline:
                reason = "fit deadline"
                break
            if time.monotonic() - best_at > max(30.0, 0.3 * seconds):
                reason = "training loss stagnation"
                break
            # Adam, SGD and LBFGS all implement this native PyTorch closure interface.
            optimizer.step(closure)
            completed_steps += 1
        # The retained state is exactly one state whose private loss was evaluated.
        model.load_state_dict(best_state)
    displacement = sum(float((best_state[k] - initial_state[k]).square().sum())
                       for k in initial_state if initial_state[k].is_floating_point()) ** 0.5
    save_artifacts(model, config, log, history)
    return {"optimizer_steps_completed": completed_steps, "closure_calls": closures,
            "fit_seconds": time.monotonic() - start, "fit_cpu_seconds": time.process_time() - cpu_start,
            "stop_reason": reason,
            "best_training_loss": best_loss if closures else None,
            "state_displacement": displacement,
            "trainable_parameters": sum(p.numel() for p in trainable)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=os.environ.get("EVE_SOLVER_ROOT"))
    parser.add_argument("--log", type=Path, default=os.environ.get("EVE_EVAL_LOG_ROOT"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--fit", action="store_true", help="Fit rather than replay a workspace checkpoint")
    parser.add_argument("--start-state", type=Path, help="Own-session state.pt to warm-start fitting or score")
    parser.add_argument("--score-only", action="store_true", help="Score the supplied state without fitting")
    parser.add_argument("--effective-config", type=Path, help="Own frozen config for score-only reconstruction")
    parser.add_argument("--checkpoint-out", type=Path, help="Export the selected state through native state/ artifacts")
    args = parser.parse_args()
    if args.workspace is None or args.log is None:
        parser.error("Native workspace and evaluation log paths are required")
    args.log.mkdir(parents=True, exist_ok=True)
    args.workspace = args.workspace.resolve()
    args.log = args.log.resolve()
    score = dict(FAILURE)
    try:
        threads = int(os.environ["ASYS_NUM_THREADS"])
        seconds = float(os.environ["ASYS_WALL_TIME"])
        if threads < 1 or seconds <= 0:
            raise ValueError("Declare positive thread and fit-time allowances")
        torch.set_num_threads(threads)
        overrides = {k.replace("-", "_"): v for k, v in
                     (("seed", args.seed),
                      ("max_steps", args.max_steps)) if v is not None}
        if args.effective_config:
            if not args.score_only or overrides:
                raise ValueError("Frozen effective config requires score-only without CLI overrides")
            overrides = yaml.safe_load(args.effective_config.read_text())
        started = time.monotonic()
        cpu_started = time.process_time()
        source_bytes = {name: (args.workspace / name).read_bytes() for name in SOURCE_FILES}
        source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()}
        (args.log / "source").mkdir()
        for name, data in source_bytes.items():
            (args.log / "source" / name).write_bytes(data)
        problem, config, model = load(args.workspace, overrides)
        checkpoint_path = args.workspace / "state" / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True) if checkpoint_path.exists() else None
        replay = checkpoint is not None and not args.fit and not args.score_only
        if replay:
            if args.start_state or overrides:
                raise ValueError("Checkpoint replay cannot change source configuration or initial state")
            if checkpoint["source_sha256"] != source_hashes or checkpoint["effective_config"] != config:
                raise ValueError("Checkpoint source/config changed; refit and export a new checkpoint")
            model.load_state_dict(checkpoint["state"])
        elif args.start_state:
            model.load_state_dict(torch.load(args.start_state, weights_only=True))
        if replay or args.score_only:
            save_artifacts(model, config, args.log, [])
            fit_info = {"optimizer_steps_completed": 0, "closure_calls": 0, "fit_seconds": 0.0, "fit_cpu_seconds": 0.0,
                        "stop_reason": "checkpoint replay" if replay else "score only",
                        "best_training_loss": None, "state_displacement": 0.0,
                        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        else:
            fit_info = fit(problem, config, model, args.workspace, args.log, seconds)
        if any((args.workspace / name).read_bytes() != data or
               (args.log / "source" / name).read_bytes() != data for name, data in source_bytes.items()):
            raise ValueError("Source changed before frozen scoring")
        state_hash = hashlib.sha256((args.log / "state.pt").read_bytes()).hexdigest()
        if args.score_only:
            # This fresh CLI reconstruction never imports the fitting loss/optimizer.
            scoring_problem = module("_ac_public_problem", Path(__file__).with_name("problem.py"))
            score = public_score(scoring_problem, model)
            scoring_info = {"public_scoring_contract": "score-only source/config/state reconstruction",
                            "scoring_optimizer_steps": 0, "scoring_cpu_seconds": 0.0}
        else:
            score, scoring_info = frozen_public_score(args.log)
        if any((args.workspace / name).read_bytes() != data or
               (args.log / "source" / name).read_bytes() != data for name, data in source_bytes.items()):
            raise ValueError("Source changed during frozen scoring")
        if hashlib.sha256((args.log / "state.pt").read_bytes()).hexdigest() != state_hash:
            raise ValueError("Persisted state changed during frozen scoring")
        manifest = {"source_sha256": source_hashes,
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "state_sha256": state_hash,
            "training_history_sha256": hashlib.sha256(
                (args.log / "training_history.pt").read_bytes()).hexdigest(),
            "effective_config_sha256": hashlib.sha256(
                (args.log / "effective_config.yaml").read_bytes()).hexdigest(),
            "torch_version": str(torch.__version__), "numerical_threads": threads,
            "fit_allowance_seconds": seconds, "total_seconds": time.monotonic() - started,
            "total_cpu_seconds": time.process_time() - cpu_started + scoring_info["scoring_cpu_seconds"],
            "process_id": os.getpid(), "parent_process_id": os.getppid(),
            "operation": "checkpoint_replay" if replay else "score_only" if args.score_only
                         else "warm_start_fit" if args.start_state else "constructor_fit",
            "selected_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() if replay else None,
            "start_state_sha256": hashlib.sha256(args.start_state.read_bytes()).hexdigest() if args.start_state else None,
            **fit_info, **scoring_info}
        (args.log / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if args.checkpoint_out:
            raw_config = yaml.safe_load((args.workspace / "config.yaml").read_text())
            if config != raw_config:
                raise ValueError("An eligible native checkpoint must use the workspace config without CLI overrides")
            payload = {"source_sha256": source_hashes, "effective_config": config,
                       "state": torch.load(args.log / "state.pt", weights_only=True),
                       "state_sha256": manifest["state_sha256"],
                       "public_score": score, "fit_manifest": manifest}
            args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, args.checkpoint_out)
    except Exception:
        score = {**FAILURE, "summary": "Allen-Cahn candidate failure"}
        (args.log / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.log / "score.yaml").write_text(yaml.safe_dump(score, sort_keys=False))
    print(json.dumps(score))


if __name__ == "__main__":
    main()
