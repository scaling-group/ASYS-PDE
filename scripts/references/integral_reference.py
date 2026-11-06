"""Integral reference operators from arXiv:2305.05895v5, Section 3.1.

F is even, F(0)=1, F''(0)=-2. Supply log(F(exp(w))) to retain the
small origin correction and avoid subtracting a constant from the velocity.
"""

import math

from scipy.integrate import quad


def kernel(log_ratio):
    """r log|(1+r)/(1-r)| - 2, without large-r cancellation."""
    if log_ratio > 1:
        z = math.exp(-2 * log_ratio)
        power, total = z, 0.0
        for k in range(1, 32):
            term = 2 * power / (2 * k + 1)
            total += term
            if term < 1e-17 * total:
                break
            power *= z
        return total
    distance = abs(log_ratio)
    if distance == 0:
        raise ValueError("The logarithmic singularity must be an integration endpoint")
    logarithm = math.log1p(math.exp(-distance)) - math.log(-math.expm1(-distance))
    return math.exp(log_ratio) * logarithm - 2


class IntegralOperators:
    """Adaptive log-radius quadrature with explicit integration refinement."""

    def __init__(
        self,
        log_profile,
        extent=40.0,
        tolerance=1e-11,
        panel_width=None,
        singularity_rule="adaptive",
    ):
        self.log_profile = log_profile
        self.extent = float(extent)
        self.tolerance = float(tolerance)
        self.panel_width = None if panel_width is None else float(panel_width)
        self.singularity_rule = singularity_rule
        if singularity_rule not in {"adaptive", "weighted_log"}:
            raise ValueError("Unknown logarithmic singularity rule")
        if self.panel_width is not None and self.panel_width <= 0:
            raise ValueError("Panel width must be positive")
        self.b = 2 / math.pi * self.integrate(lambda w: math.exp(self.log_profile(w) + w))
        self.c = (
            2 / math.pi * self.integrate(lambda w: -math.expm1(self.log_profile(w)) * math.exp(-w))
        )
        if not (self.b > 0 and self.c > 0):
            raise ValueError("Positive finite moments are required")

    def integrate(self, function, singular=None, upper=None, lower=None, absolute_tolerance=None):
        start = -self.extent if lower is None else float(lower)
        end = self.extent if upper is None else float(upper)
        if not -self.extent <= start < end <= self.extent:
            raise ValueError("Integration limit outside declared log-radius interval")
        points = [start, end]
        points.extend(x for x in (0.0, singular) if x is not None and start < x < end)
        if self.panel_width is not None:
            # Resolve localized kernel features before QUADPACK extrapolates a
            # long interval. Splitting retains the total absolute error budget.
            count = math.ceil((end - start) / self.panel_width)
            points.extend(start + k * (end - start) / count for k in range(1, count))
            if singular is not None:
                points.extend(x for x in (singular - 1, singular + 1) if start < x < end)
        points = sorted(set(points))
        tolerance = self.tolerance if absolute_tolerance is None else absolute_tolerance
        absolute_budget = (
            tolerance / (len(points) - 1) if self.panel_width is not None else tolerance
        )
        return sum(
            quad(function, lo, hi, epsabs=absolute_budget, epsrel=self.tolerance, limit=300)[0]
            for lo, hi in zip(points[:-1], points[1:], strict=True)
        )

    def transform_scaled(self, log_x):
        """Return T(F)(x)/min(x*x, 1), preserving the origin limit -c/3."""
        subtract_origin = log_x <= 0
        scale = math.exp(2 * log_x) if subtract_origin else 1.0

        def integrand(w):
            log_f = self.log_profile(w)
            value = math.expm1(log_f) if subtract_origin else math.exp(log_f)
            return value * math.exp(w) * kernel(w - log_x) / scale

        if self.singularity_rule == "weighted_log":
            return self.weighted_transform(log_x, scale, subtract_origin, integrand) / math.pi

        # The kernel's integral against the constant 1 is zero on (0, infinity).
        # Subtracting F(0) avoids loss of the O(x^2) signal at small x.
        return self.integrate(integrand, singular=log_x) / math.pi

    def weighted_transform(self, log_x, scale, subtract_origin, integrand):
        """Resolve log|w-log(x)| analytically with QUADPACK endpoint weights.

        For t=w-log(x), K(exp(t)) = exp(t)*(L(t)-log|t|)-2,
        where L(t)=log(|t|/tanh(|t|/2)) is smooth and L(0)=log(2).
        """
        if not -self.extent < log_x < self.extent:
            raise ValueError("Transform site must lie inside the integration interval")
        left, right = max(-self.extent, log_x - 1), min(self.extent, log_x + 1)
        budget = self.tolerance / 5

        def factor(w):
            log_f = self.log_profile(w)
            f = math.expm1(log_f) if subtract_origin else math.exp(log_f)
            return f * math.exp(w) / scale

        def regular(w):
            t = w - log_x
            u = abs(t)
            smooth = math.log(u / math.tanh(u / 2)) if u else math.log(2)
            return factor(w) * (math.exp(t) * smooth - 2)

        def coefficient(w):
            return factor(w) * math.exp(w - log_x)

        result = self.integrate(regular, lower=left, upper=right, absolute_tolerance=budget)
        for lo, hi, weight in ((left, log_x, "alg-logb"), (log_x, right, "alg-loga")):
            result -= quad(
                coefficient,
                lo,
                hi,
                weight=weight,
                wvar=(0.0, 0.0),
                epsabs=budget,
                epsrel=self.tolerance,
                limit=300,
            )[0]
        if left > -self.extent:
            result += self.integrate(integrand, upper=left, absolute_tolerance=budget)
        if right < self.extent:
            result += self.integrate(integrand, lower=right, absolute_tolerance=budget)
        return result

    def transform(self, log_x):
        return self.transform_scaled(log_x) * min(math.exp(2 * log_x), 1.0)

    def log_map(self, a, log_x):
        """Published nonlinear map, evaluated by nested adaptive integration."""
        if a == 0:
            integral = self.integrate(self.transform, upper=log_x)
            return 2 / self.c * (self.transform(log_x) + integral)

        factor = 2 * a / ((1 - a / 3) * self.c)

        def correction(w):
            delta = factor * self.transform(w)
            if delta <= -1:
                raise ValueError("Nonpositive G: this implementation requires full support")
            return delta / (1 + delta)

        delta = factor * self.transform(log_x)
        if delta <= -1:
            raise ValueError("Nonpositive G")
        integral = self.integrate(correction, upper=log_x)
        return math.log1p(delta) / a + (1 - a) / a * integral

    def constants(self, a):
        c_l = (1 - a / 3) * self.c / 2 - a * self.b
        c_omega = (1 - a / 3) * self.c / 2 - self.b
        if c_l <= 0 or c_omega >= 0:
            raise ValueError("Expected a focusing, decaying full-support profile")
        return dict(
            b=self.b,
            c=self.c,
            c_l=c_l,
            c_omega=c_omega,
            beta=c_l / -c_omega,
            amplitude=self.b / -c_omega,
        )
