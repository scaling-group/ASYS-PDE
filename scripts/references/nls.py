"""Periodic NLS reference with spatial and temporal refinement checks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

T_MAX = math.pi / 2
LEVELS = ((4096, 512), (8192, 512), (4096, 1024))


def weak_times(nt, order, offset):
    h = T_MAX / (nt + 1)
    a = (np.arange(nt) + 0.25 + 0.5 * offset) * h
    nodes, _ = np.polynomial.legendre.leggauss(order)
    return np.concatenate((a, a + h, (a[:, None] + h * (nodes[None, :] + 1) / 2).ravel()))


def schedule():
    times = [np.linspace(0, T_MAX, 81)]
    for nt, order in ((41, 8), (41, 16), (82, 16)):
        for offset in (0.0, 0.5):
            times.append(weak_times(nt, order, offset))
    return np.unique(np.concatenate(times))


def strang(width, nx, substeps, times, initial=None):
    x = np.arange(nx) * 2 * width / nx - width
    z = np.asarray(2 / np.cosh(x) if initial is None else initial(x), dtype=np.complex128)
    k = 2 * np.pi * np.fft.fftfreq(nx, d=2 * width / nx)
    dt_max = T_MAX / (80 * substeps)
    current, states = 0.0, []
    for target in times:
        while target - current > 1e-14:
            dt = min(dt_max, target - current)
            half = np.exp(-0.25j * k * k * dt)
            z = np.fft.ifft(np.fft.fft(z) * half)
            z *= np.exp(1j * np.abs(z) ** 2 * dt)
            z = np.fft.ifft(np.fft.fft(z) * half)
            current += dt
        states.append(z.copy())
    return x, np.asarray(states)


def breather(x, times):
    xx, tt = x[None, :], times[:, None]
    return (
        4
        * np.exp(0.5j * tt)
        * (np.cosh(3 * xx) + 3 * np.exp(4j * tt) * np.cosh(xx))
        / (np.cosh(4 * xx) + 4 * np.cosh(2 * xx) + 3 * np.cos(4 * tt))
    )


def plane_wave():
    errors = []
    for width in (3.5, 5.0, 7.0):
        k, amplitude = 3 * math.pi / width, 0.7
        t = np.array([0.0, 0.31, T_MAX])
        x, actual = strang(
            width,
            128,
            16,
            t,
            initial=lambda x, amplitude=amplitude, k=k: amplitude * np.exp(1j * k * x),
        )
        expected = amplitude * np.exp(
            1j * (k * x[None, :] + (amplitude**2 - 0.5 * k * k) * t[:, None])
        )
        errors.append(float(np.max(np.abs(actual - expected))))
    if max(errors) > 2e-11:
        raise AssertionError("reference phase-sign/plane-wave qualification failed")
    return errors


def generate(root, width):
    root.mkdir(parents=True, exist_ok=False)
    manufactured = plane_wave()
    times = schedule()
    target = np.linspace(0, T_MAX, 161)
    right = np.searchsorted(times, target).clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    ti = np.unique(np.where(abs(times[left] - target) < abs(times[right] - target), left, right))
    assert len(ti) == 161
    common_times = times[ti]
    levels, mass_drifts, finest = {}, {}, None
    for nx, substeps in LEVELS:
        x, states = strang(width, nx, substeps, times)
        key = f"{nx}/{substeps}"
        levels[key] = states[ti, :: nx // 1024].copy()
        mass = (np.abs(states) ** 2).mean(axis=1) * 2 * width
        mass_drifts[key] = float(np.max(np.abs(mass / mass[0] - 1)))
        if (nx, substeps) == LEVELS[0]:
            finest, reference_x = states, x
    common_x = np.arange(1024) * 2 * width / 1024 - width
    truth = levels["4096/512"]
    correction = truth - breather(common_x, common_times)

    def norm(value):
        energy = np.mean(np.abs(value) ** 2, axis=1)
        return np.sqrt(np.sum(np.diff(common_times) * (energy[1:] + energy[:-1]) / 2))

    field_norm, correction_norm = norm(truth), norm(correction)
    convergence = {}
    for axis, key in (("space", "8192/512"), ("time", "4096/1024")):
        delta = norm(truth - levels[key])
        convergence[axis] = dict(
            field_relative=float(delta / field_norm),
            correction_relative=float(delta / correction_norm),
        )
    qualified = (
        all(
            r["field_relative"] <= 1e-5 and r["correction_relative"] <= 0.01
            for r in convergence.values()
        )
        and max(mass_drifts.values()) <= 1e-10
    )
    np.savez(
        root / "reference.npz",
        width=width,
        times=times,
        x=reference_x,
        states=finest,
        readout_times=common_times,
        readout_x=common_x,
        readout_state=truth,
    )
    receipt = dict(
        width=width,
        original_initial_datum="2 sech(x), periodic H1",
        levels=LEVELS,
        reference_level=(4096, 512),
        readout_shape=truth.shape,
        norm="uniform spatial mean, trapezoidal time weights on nearest saved times",
        phase_sign_errors=manufactured,
        mass_drifts=mass_drifts,
        convergence=convergence,
        field_norm=float(field_norm),
        correction_norm=float(correction_norm),
        reference_refinement_passed=qualified,
    )
    (root / "reference_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not qualified:
        raise RuntimeError("NLS reference refinement checks failed")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--width", type=float, choices=(3.5, 4.0, 4.5, 5.0, 6.0, 7.0), required=True)
    a = p.parse_args()
    print(json.dumps(generate(a.root, a.width), indent=2))
