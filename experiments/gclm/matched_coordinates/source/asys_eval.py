"""Protected native fit, generic frozen export, public assessment and replay."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "kernel"))
import numpy as np
import torch
import core_fitting
import common_single
from frozen_profile import FrozenProfile, origin_slope


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    spec = importlib.util.spec_from_file_location("native_matched_model", ROOT / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_model(dict(config)).cpu().double()


def evaluate(log):
    torch.set_num_threads(1)
    config = json.loads((ROOT / "config.json").read_text())
    config["seed"] = int(os.environ["ASYS_SEED"])
    write(log / "model_config.json", config)
    started = time.monotonic()
    raw = build(config)
    initial_beta = float(raw.beta())
    trainable_parameters = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    assert trainable_parameters == 14
    origin_slope(raw)
    policy = json.loads((ROOT / "fit_policy.json").read_text())
    if any(p.requires_grad and p.numel() for p in raw.parameters()):
        # Discard the unit-slope-specific SIGS wrapper, retaining its chosen raw
        # state and final measured mass. Mathematical fitting code is unchanged.
        _, fit = core_fitting.fit(raw, policy, "terminal")
    else:
        work = core_fitting.Work(policy["terminal"]["seconds"],
                                 policy["terminal"]["max_field_elements"])
        validation = core_fitting.Objective(policy, policy["terminal"], work).validate(raw)
        fit = dict(eligible=validation["resolved"], stop="parameterless_no_fit",
                   updates=0, closures=0, trace=[], validation=validation,
                   selected_state="unchanged_parameterless_construction", work=work.summary())
    assert float(raw.beta()) == initial_beta, "Beta changed"
    fit["fixed_beta_verified"] = initial_beta
    fit["trainable_shape_coordinates"] = 14
    write(log / "fit.json", fit)
    if not fit["eligible"]:
        raise ValueError("Fit ineligible; complete fit record retained")
    amplitude = fit["validation"]["mass"]
    slope = origin_slope(raw)
    fit["validation"]["original_unit_slope_canonical_core"] = fit["validation"]["canonical_core"]
    fit["validation"]["canonical_core"] = [float(v - np.log(amplitude / slope))
                                             for v in fit["validation"]["raw_core"]]
    write(log / "fit.json", fit)
    frozen = FrozenProfile(raw, amplitude, slope)
    normalization = dict(amplitude=amplitude, coordinate_scale=amplitude / slope,
                         origin_slope=slope, beta=float(raw.beta()),
                         canonical_core=[v - np.log(amplitude / slope)
                                         for v in fit["validation"]["raw_core"]])
    write(log / "normalization.json", normalization)
    state = log / "state_dict.pt"
    torch.save(frozen.state_dict(), state)
    # Rebuild once, strictly load exported state; never survey or regauge replay.
    replay = FrozenProfile(build(config), amplitude, slope)
    replay.load_state_dict(torch.load(state, weights_only=True, map_location="cpu"), strict=True)
    y = torch.tensor(np.r_[-np.logspace(-3, 3, 11), 0., np.logspace(-3, 3, 11)],
                     dtype=torch.float64, requires_grad=True)
    values = frozen.profile(y)
    restored = replay.profile(y)
    derivative = torch.autograd.grad(values.sum(), y)[0]
    restored_derivative = torch.autograd.grad(restored.sum(), y)[0]
    replay_pass = torch.equal(values, restored) and torch.equal(derivative, restored_derivative)
    write(log / "replay.json", dict(passed=bool(replay_pass), sites=y.detach().tolist(),
                                    exact_values=bool(torch.equal(values, restored)),
                                    exact_coordinate_derivatives=bool(torch.equal(derivative, restored_derivative))))
    if not replay_pass:
        raise ValueError("Fresh reconstruction replay differs")
    assessment_config = json.loads((ROOT / "assessment_config.json").read_text())
    budget = core_fitting.Work(120, assessment_config["max_field_elements"])
    assessment = common_single.assess_family(replay, anchors=(.25,),
                                             config=assessment_config, budget=budget)
    write(log / "public.json", assessment)
    valid = assessment["status"] == "qualified" and assessment["score"] is not None
    score = -float(assessment["score"]) if valid else -1e9
    bindings = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    write(log / "source_bindings.json", bindings)
    write(log / "state_binding.json", dict(sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
                                          bytes=state.stat().st_size))
    return dict(score=score, physics=score, normalization=0. if valid else -1e9,
                refinement=0. if valid else -1e9,
                summary="Common finite instrument " + assessment["status"],
                info=dict(public_status=assessment["status"], fit_updates=fit["updates"],
                          fit_stop=fit["stop"], elapsed_seconds=time.monotonic()-started,
                          fitted_parameters=trainable_parameters,
                          stored_raw_parameters=sum(p.numel() for p in raw.parameters()),
                          buffer_scalars=sum(p.numel() for p in frozen.buffers()),
                          normalization=normalization, reference_access=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate(args.log)
    except Exception as error:
        result = dict(score=-1e9, physics=-1e9, normalization=-1e9, refinement=-1e9,
                      summary=type(error).__name__ + ": " + str(error))
        write(args.log / "failure.json", result)
    write(args.log / "score.yaml", result)
