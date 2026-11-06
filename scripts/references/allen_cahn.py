"""u_t = epsilon^2 Delta u + u - u^3 on [-1,1)^2, epsilon=0.04.

ETDRK4: https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.fft import fft2, ifft2, irfft2, rfft2

EPSILON = 0.04


def initial_condition(x, y, delta0=1e-14):
    scale = np.sqrt(2) * EPSILON
    d1 = np.sqrt((x + 0.15) ** 2 + (y - 0.02) ** 2 + delta0)
    d2 = np.sqrt((x - 0.18) ** 2 + (y + 0.03) ** 2 + delta0)
    p1 = np.tanh((0.22 - d1) / scale)
    p2 = np.tanh((0.18 - d2) / scale)
    return 0.5 * (p1 + p2 + 1 - p1 * p2)


def reaction(value, dt):
    """Exact scalar flow of u'=u-u^3, with its sign preserved."""
    return value * np.exp(dt) / np.sqrt(1 + value**2 * np.expm1(2 * dt))


def diffuse(value, multiplier, workers):
    return ifft2(multiplier * fft2(value, workers=workers), workers=workers).real


def advance(value, half_multiplier, dt, workers):
    value = diffuse(value, half_multiplier, workers)
    value = reaction(value, dt)
    return diffuse(value, half_multiplier, workers)


def grid(cells):
    x = -1 + 2 * np.arange(cells) / cells
    frequency = 2 * np.pi * np.fft.fftfreq(cells, d=2 / cells)
    return x, frequency[:, None] ** 2 + frequency[None, :] ** 2


def energy(value, frequency_squared, workers):
    coefficients = fft2(value, norm="forward", workers=workers)
    return float(
        4
        * (
            0.5 * EPSILON**2 * np.sum(frequency_squared * np.abs(coefficients) ** 2)
            + 0.25 * np.mean((1 - value**2) ** 2)
        )
    )


def integrate(cells, times, maximum_dt, workers):
    if maximum_dt <= 0 or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Need positive maximum step and increasing nonnegative snapshot times")
    x, frequency_squared = grid(cells)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    value = initial_condition(xx, yy)
    snapshots, energies = [], []
    initial_energy = energy(value, frequency_squared, workers)
    current, steps = 0.0, 0
    minimum, maximum = float(value.min()), float(value.max())
    started = time.monotonic()
    for snapshot_time in times:
        interval = float(snapshot_time - current)
        if interval > 0:
            count = max(1, math.ceil(interval / maximum_dt))
            dt = interval / count
            half = np.exp(-0.5 * EPSILON**2 * frequency_squared * dt)
            for _ in range(count):
                value = advance(value, half, dt, workers)
            steps += count
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite trajectory")
        snapshots.append(value.copy())
        energies.append(energy(value, frequency_squared, workers))
        minimum, maximum = min(minimum, float(value.min())), max(maximum, float(value.max()))
        current = float(snapshot_time)
    energy_sequence = np.array([initial_energy, *energies])
    return (
        x,
        np.array(snapshots),
        {
            "cells_per_axis": cells,
            "maximum_dt": maximum_dt,
            "total_steps": steps,
            "integration_and_diagnostics_seconds": time.monotonic() - started,
            "initial_energy": initial_energy,
            "snapshot_energies": energies,
            "maximum_snapshot_energy_increase": float(max(0, np.diff(energy_sequence).max())),
            "snapshot_minimum": minimum,
            "snapshot_maximum": maximum,
        },
    )


def calibration(workers):
    cells, dt = 32, 0.137
    x, k2 = grid(cells)
    half = np.exp(-0.5 * EPSILON**2 * k2 * dt)
    constant_error = 0.0
    for amplitude in (-0.9, 0.0, 0.4, 1.0):
        initial = np.full((cells, cells), amplitude)
        actual = advance(initial, half, dt, workers)
        exact = amplitude / np.sqrt(amplitude**2 + (1 - amplitude**2) * np.exp(-2 * dt))
        constant_error = max(constant_error, float(np.max(np.abs(actual - exact))))
    xx, yy = np.meshgrid(x, x, indexing="ij")
    mode = np.cos(2 * np.pi * xx) * np.cos(3 * np.pi * yy)
    actual = diffuse(mode, np.exp(-(EPSILON**2) * k2 * dt), workers)
    exact = mode * np.exp(-(EPSILON**2) * 13 * np.pi**2 * dt)
    diffusion_error = float(np.max(np.abs(actual - exact)))
    if constant_error > 1e-12 or diffusion_error > 1e-12:
        raise ValueError("Exact-flow calibration failed")
    return {
        "constant_solution_max_error": constant_error,
        "diffusion_mode_max_error": diffusion_error,
    }


def integrate_etdrk4(n, times, maximum_dt, initial=None):
    if maximum_dt <= 0 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Need positive maximum step and increasing times starting at zero")
    started = time.monotonic()
    x = -1 + 2 * np.arange(n) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u = initial_condition(xx, yy, 1e-12) if initial is None else initial(xx, yy)
    v = u + 1
    kx = np.fft.fftfreq(n, d=2 / n) * 2 * np.pi
    ky = np.fft.rfftfreq(n, d=2 / n) * 2 * np.pi
    linear = -(EPSILON**2) * (kx[:, None] ** 2 + ky[None, :] ** 2) - 2
    coeff_cache = {}

    def coefficients(dt):
        key = round(dt, 15)
        if key not in coeff_cache:
            z = dt * linear
            circle = np.exp(1j * np.pi * (np.arange(32) + 0.5) / 32)
            Q = np.zeros_like(z)
            f1, f2, f3 = Q.copy(), Q.copy(), Q.copy()
            for r in circle:
                a = z + r
                ex = np.exp(a)
                Q += ((np.exp(a / 2) - 1) / a).real / 32
                f1 += ((-4 - a + ex * (4 - 3 * a + a * a)) / a**3).real / 32
                f2 += ((2 + a + ex * (-2 + a)) / a**3).real / 32
                f3 += ((-4 - 3 * a - a * a + ex * (4 - a)) / a**3).real / 32
            coeff_cache[key] = (np.exp(z), np.exp(z / 2), dt * Q, dt * f1, dt * f2, dt * f3)
        return coeff_cache[key]

    def nonlin(f):
        val = irfft2(f, s=(n, n), workers=1)
        return rfft2(val * val * (3 - val), workers=1)

    f = rfft2(v, workers=1)
    steps, snapshots = 0, []
    for it, target in enumerate(times):
        if it:
            interval = target - times[it - 1]
            count = int(np.ceil(interval / maximum_dt - 1e-10))
            dt = interval / count
            E, E2, Q, f1, f2, f3 = coefficients(dt)
            for _ in range(count):
                nv = nonlin(f)
                a = E2 * f + Q * nv
                na = nonlin(a)
                b = E2 * f + Q * na
                nb = nonlin(b)
                c = E2 * a + Q * (2 * nb - nv)
                nc = nonlin(c)
                f = E * f + f1 * nv + 2 * f2 * (na + nb) + f3 * nc
            steps += count
        value = irfft2(f, s=(n, n), workers=1) - 1
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite ETDRK4 field")
        snapshots.append(value)
    return (
        x,
        np.asarray(snapshots),
        dict(
            method="ETDRK4",
            cells=n,
            maximum_dt=maximum_dt,
            steps=steps,
            contour_points=32,
            initial_regularizer=1e-12 if initial is None else None,
            time_knots=len(times),
            elapsed_seconds=time.monotonic() - started,
        ),
    )


def calibration_etdrk4():
    errors = []
    times = np.array([0.0, 0.25])
    for dt in (1 / 16, 1 / 32):
        _, values, _ = integrate_etdrk4(8, times, dt, lambda x, y: np.full_like(x, 0.4))
        errors.append(float(np.max(np.abs(values[-1] - reaction(0.4, 0.25)))))
    if errors[1] > 1e-7 or errors[0] < 8 * errors[1]:
        raise ValueError("ETDRK4 reaction-flow calibration failed")
    return errors
