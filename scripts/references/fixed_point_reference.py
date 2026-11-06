"""Numerical log-radius representation for the qualified integral reference."""

import math
import time

import numpy as np
from integral_reference import IntegralOperators
from scipy.interpolate import make_interp_spline


class LogProfile:
    """Quintic interpolation of log F, with analytic origin and power tail."""

    def __init__(self, w, log_values, tail_power):
        self.w = np.asarray(w, dtype=float)
        self.log_values = np.asarray(log_values, dtype=float)
        self.tail_power = float(tail_power)
        if (
            not np.isfinite(self.log_values).all()
            or self.tail_power <= 1
            or not np.all(np.diff(self.w) > 0)
        ):
            raise ValueError("Invalid positive, integrable profile representation")
        self.spline = make_interp_spline(self.w, self.log_values, k=5)
        self.slope = self.spline.derivative()

    def __call__(self, w):
        if w < self.w[0]:
            return -math.exp(2 * w)
        if w > self.w[-1]:
            return self.log_values[-1] - self.tail_power * (w - self.w[-1])
        return float(self.spline(w))

    def derivative(self, w):
        if w < self.w[0]:
            return -2 * math.exp(2 * w)
        if w > self.w[-1]:
            return -self.tail_power
        return float(self.slope(w))


class CanonicalProfile:
    """f(y)=y F(b*y), with f'(0)=g(0)=1 in the exact moment limit."""

    def __init__(self, profile, b):
        self.profile, self.b = profile, float(b)

    def value(self, y):
        if y == 0:
            return 0.0
        return y * math.exp(self.profile(math.log(abs(self.b * y))))

    def derivative(self, y):
        if y == 0:
            return 1.0
        w = math.log(abs(self.b * y))
        return math.exp(self.profile(w)) * (1 + self.profile.derivative(w))

    def evaluate(self, y):
        return (
            np.array([self.value(float(x)) for x in y]),
            np.array([self.derivative(float(x)) for x in y]),
        )


def operators(profile, config):
    return IntegralOperators(
        profile,
        config["integration_extent"],
        config["integration_tolerance"],
        config["panel_width"],
        config.get("singularity_rule", "adaptive"),
    )


def map_step(profile, a, config, deadline=math.inf):
    """Evaluate T at the nodes, then integrate a quintic spline of the map field."""
    op = operators(profile, config)
    transform = []
    for w in profile.w:
        if time.monotonic() >= deadline:
            raise TimeoutError("Numerical iteration exceeded its declared time budget")
        transform.append(op.transform(float(w)))
    transform = np.asarray(transform)
    if a == 0:
        field = transform
        tail_power = 2 * op.b / op.c
    else:
        factor = 2 * a / ((1 - a / 3) * op.c)
        delta = factor * transform
        g_infinity = 1 - factor * op.b
        if min(float(np.min(1 + delta)), g_infinity) <= 0:
            raise ValueError("Iteration left the declared full-support branch")
        field = delta / (1 + delta)
        tail_power = (1 - a) / a * (1 / g_infinity - 1)
    primitive = make_interp_spline(profile.w, field, k=5).antiderivative()
    # At the origin the field is proportional to exp(2w); its omitted integral
    # is field(w_min)/2 up to O(exp(4w_min)), far below the stated tolerances.
    integrated = primitive(profile.w) - primitive(profile.w[0]) + field[0] / 2
    log_values = (
        2 / op.c * (transform + integrated)
        if a == 0
        else np.log1p(delta) / a + (1 - a) / a * integrated
    )
    return LogProfile(profile.w, log_values, tail_power), op


def iterate(profile, a, config, progress=None):
    started = time.monotonic()
    deadline = started + config["maximum_seconds"]
    trace = []
    for iteration in range(1, config["maximum_iterations"] + 1):
        updated, op = map_step(profile, a, config, deadline)
        new_op = operators(updated, config)
        before, after = op.constants(a), new_op.constants(a)
        change = float(np.max(abs(updated.log_values - profile.log_values)))
        beta_change = abs(after["beta"] - before["beta"])
        tail_change = abs(updated.tail_power - (1 + 1 / after["beta"]))
        trace.append(
            dict(
                iteration=iteration,
                maximum_log_profile_update=change,
                beta=after["beta"],
                beta_change=beta_change,
                tail_power=updated.tail_power,
                tail_consistency_error=tail_change,
                elapsed_seconds=time.monotonic() - started,
            )
        )
        profile = updated
        if progress:
            progress(trace)
        if max(change, beta_change, tail_change) < config["iteration_tolerance"]:
            return profile, dict(
                converged=True,
                trace=trace,
                constants=after,
                elapsed_seconds=time.monotonic() - started,
            )
    return profile, dict(
        converged=False, trace=trace, constants=after, elapsed_seconds=time.monotonic() - started
    )
