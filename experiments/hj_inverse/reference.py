"""Regenerate and qualify the HJ researcher reference with the original Fourier/DOP853 recipe."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent


def rollout(coefficients, problem, times, n):
    axis = np.arange(n) * 2 * np.pi / n
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    initial = np.zeros((n, n))
    for kx, ky, amplitude, phase in problem["initial_modes"]:
        initial += amplitude * np.sin(kx * xx + ky * yy + phase) / (abs(kx) + abs(ky) + 1)
    initial /= problem["initial_scale"]
    wave = np.fft.fftfreq(n, d=1 / n)
    kx, ky = wave[None], wave[:, None]
    dx, dy, bx, by = coefficients

    def rhs(t, state):
        spectrum = np.fft.fft2(state.reshape(n, n))
        ux = np.fft.ifft2(1j * kx * spectrum).real
        uy = np.fft.ifft2(1j * ky * spectrum).real
        return (np.fft.ifft2(-(dx * kx**2 + dy * ky**2) * spectrum).real + bx * ux**2 + by * uy**2).ravel()

    result = solve_ivp(rhs, (0, float(max(times))), initial.ravel(), t_eval=times,
                       method="DOP853", rtol=1e-9, atol=1e-11, max_step=0.001)
    if not result.success or not np.isfinite(result.y).all():
        raise ValueError("Reference PDE solve failed")
    return result.y.T.reshape(len(times), n, n)



def generate(output):
    if output.exists():
        raise FileExistsError(f"Choose a fresh reference directory: {output}")
    problem = json.loads((ROOT / "polynomial/problem.json").read_text())
    metadata = json.loads((ROOT / "researcher_reference/assessment.json").read_text())
    coefficients = np.asarray(metadata["coefficient_truth"])
    times = np.asarray(metadata["times"])
    coarse = rollout(coefficients, problem, times, 128)
    fine = rollout(coefficients, problem, times, 256)
    sampled = fine[:, ::2, ::2]
    errors = (np.linalg.norm((coarse - sampled).reshape(50, -1), axis=1)
              / np.linalg.norm(sampled.reshape(50, -1), axis=1))
    maximum = float(errors.max())
    if not np.isfinite(maximum) or maximum > metadata["qualification"]["tolerance"]:
        raise ValueError(f"Grid qualification failed: {maximum}")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "truth.npz"
    np.savez(target, u=fine[:, ::4, ::4], coefficients=coefficients)
    metadata["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata["qualification"]["max_grid_relative_l2"] = maximum
    (output / "assessment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata["qualification"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2, allow_nan=False))
