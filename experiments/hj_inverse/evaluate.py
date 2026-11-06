"""Replay the two frozen HJ fields and physical coefficients without fitting."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(arm, expected):
    folder = ROOT / arm
    for name, key in (("model.py", "model_sha256"),
                      ("problem.json", "problem_sha256"),
                      ("selected-state.pt", "selected_state_sha256")):
        if digest(folder / name) != expected[key]:
            raise ValueError(f"Frozen {arm}/{name} hash differs")
    problem = json.loads((folder / "problem.json").read_text())
    spec = importlib.util.spec_from_file_location("hj_" + arm, folder / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = module.Candidate(problem).double()
    state = torch.load(folder / "selected-state.pt", map_location="cpu", weights_only=True)
    required = {"raw", "lower", "span"} | {"candidate." + k for k in candidate.state_dict()}
    if set(state) != required or not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Invalid frozen field state")
    bounds = torch.tensor(problem["coefficient_bounds"], dtype=torch.float64)
    if not torch.equal(state["lower"], bounds[:, 0]) or not torch.equal(
        state["span"], bounds[:, 1] - bounds[:, 0]
    ):
        raise ValueError("Checkpoint coefficient bounds differ from the task")
    candidate.load_state_dict({k.removeprefix("candidate."): v for k, v in state.items()
                               if k.startswith("candidate.")}, strict=True)
    coefficients = state["lower"] + state["span"] * state["raw"].sigmoid()
    return candidate.eval(), coefficients, problem


def field_metrics(candidate, coefficients, times, truth, horizon):
    axis = np.arange(64) * 2 * np.pi / 64
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    errors, energies = [], []
    with torch.no_grad():
        for index, time in enumerate(times):
            points = np.column_stack((xx.ravel(), yy.ravel(), np.full(4096, time)))
            blocks = [candidate.forward_with_coefficients(torch.tensor(p, dtype=torch.float64),
                                                          coefficients).numpy()
                      for p in np.array_split(points, 8)]
            values = np.concatenate(blocks)
            if values.shape != (4096, 1) or not np.isfinite(values).all():
                raise ValueError("Invalid frozen field prediction")
            errors.append(float(np.square(values.reshape(64, 64) - truth[index]).sum()))
            energies.append(float(np.square(truth[index]).sum()))
    errors, energies = np.asarray(errors), np.asarray(energies)
    if not np.all(energies > 0):
        raise ValueError("Reference frame has zero energy")
    fit = times <= horizon + 1e-12
    if fit.sum() != 35 or (~fit).sum() != 15:
        raise ValueError("Expected fit frames 0–34 and future frames 35–49")
    metrics = {name: float(np.sqrt(errors[mask].sum() / energies[mask].sum()))
               for name, mask in (("fit", fit), ("future", ~fit),
                                  ("all", np.ones(len(times), dtype=bool)))}
    metrics["endpoint"] = float(np.sqrt(errors[-1] / energies[-1]))
    return metrics, np.sqrt(errors / energies).tolist()


def evaluate(reference_root=ROOT / "researcher_reference"):
    expected = json.loads((ROOT / "expected.json").read_text())
    metadata = json.loads((reference_root / "assessment.json").read_text())
    reference = reference_root / "truth.npz"
    if digest(reference) != metadata["sha256"]:
        raise ValueError("Reference hash differs")
    times = np.asarray(metadata["times"])
    with np.load(reference, allow_pickle=False) as data:
        truth, coefficients = data["u"], data["coefficients"]
    if (truth.shape != (50, 64, 64) or times.shape != (50,)
            or not np.isfinite(truth).all()
            or not np.allclose(times, np.arange(50) / 98, rtol=0, atol=1e-16)
            or not np.array_equal(coefficients, [.115, .085, -.43, -.61])
            or not np.array_equal(coefficients, metadata["coefficient_truth"])):
        raise ValueError("Unexpected reference grid, times or coefficient truth")
    rows = {}
    for arm in ("polynomial", "exponential"):
        candidate, fitted, problem = load(arm, expected[arm])
        if (problem["initial_modes"] != metadata["initial_modes"]
                or problem["initial_scale"] != metadata["initial_scale"]):
            raise ValueError("Reference and candidate initial conditions differ")
        metrics, per_frame = field_metrics(candidate, fitted, times, truth, problem["horizon"])
        relative = np.abs((fitted.numpy() - coefficients) / coefficients)
        rows[arm] = dict(
            candidate=expected[arm]["candidate"],
            selected_state_sha256=expected[arm]["selected_state_sha256"],
            model_sha256=expected[arm]["model_sha256"],
            fitted_coefficients=fitted.tolist(),
            relative_coefficient_errors=relative.tolist(),
            mean_relative_coefficient_error=float(relative.mean()),
            direct_field_relative_l2=metrics, per_frame_relative_l2=per_frame,
        )
    return dict(
        scope="Frozen expression and coefficient assessment; no fitting or inferred-PDE rollout.",
        units="Relative errors are fractions; frames and spatial nodes have equal weight.",
        coefficient_truth=coefficients.tolist(), times=times.tolist(),
        reference_sha256=metadata["sha256"],
        reference_qualification=metadata["qualification"], rows=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=ROOT / "researcher_reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    result = evaluate(args.reference_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
