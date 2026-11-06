"""Task-local support-aware surrogate, independent of the common assessor.

Fit in raw radius x. For unit raw slope h'(0)=1, with
M=(2/pi) integral h(exp(t)) dt, canonical radius
y=x/M, f=h/M, D=x*h_x/M, g=g_h/M and W=V_h/(M*x). Multiplying all four
equation terms by M preserves their signed relative residual. This avoids a
fixed canonical window and retains gradients through M. No denominator floor.
"""
import copy
import math
import os
import time
import numpy as np
import torch
from operators import Operators
from single_anchor import ANCHOR, FrozenProfile


class FitUnresolved(ValueError):
    pass


class Work:
    def __init__(self, seconds, field_limit):
        self.start = time.monotonic()
        self.deadline = self.start + seconds
        self.limit, self.fields, self.calls = field_limit, 0, 0

    def check(self, count=0):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Fit wall ceiling")
        if self.fields + count > self.limit:
            raise FitUnresolved("Fit field ceiling")
        self.fields += count
        self.calls += bool(count)

    def summary(self):
        return dict(field_elements=self.fields, field_calls=self.calls,
                    elapsed_seconds=time.monotonic()-self.start)


class CountedRaw:
    def __init__(self, raw, work):
        self.raw, self.work = raw, work

    def __call__(self, x):
        self.work.check(x.numel())
        z = self.raw(x)
        if not bool(torch.isfinite(z).all()) or bool((z < 0).any()):
            raise FitUnresolved("Nonfinite or negative raw profile")
        return z


def trapezoid_weights(t):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("Increasing integration sites required")
    w = np.zeros_like(t)
    w[:-1] += np.diff(t)/2
    w[1:] += np.diff(t)/2
    return w


def survey(raw, cfg, work):
    """Finite raw-mass survey; undershoot cannot terminate before finding a core."""
    counted = CountedRaw(raw, work)
    records, previous = [], None
    for extent in cfg["extents"]:
        converged = False
        for level in range(cfg["refinements"]):
            step = cfg["step"] / 2**level
            edges = np.linspace(-extent, extent, int(math.ceil(2*extent/step))+1)
            fine = np.linspace(-extent, extent, 2*(len(edges)-1)+1)
            mid = (edges[:-1]+edges[1:])/2
            def values(t):
                with torch.no_grad():
                    return counted(torch.tensor(np.exp(t), dtype=torch.float64)).numpy()
            f, coarse, middle = values(fine), values(edges), values(mid)
            w = trapezoid_weights(fine)*(2/math.pi)
            mass = float(np.dot(w, f))
            if not math.isfinite(mass) or mass < 0:
                raise FitUnresolved("Nonfinite raw mass")
            coarse_mass = float(np.dot(trapezoid_weights(edges)*(2/math.pi), coarse))
            phase_mass = float(np.dot(np.diff(edges)*(2/math.pi), middle))
            relative_mesh = max(abs(coarse_mass-mass), abs(phase_mass-mass))/mass if mass else None
            if mass > 0 and relative_mesh <= cfg["mass_relative_tolerance"]:
                converged = True
                break
        if mass == 0:
            records.append(dict(extent=extent, mass=0., mesh=None, status="no_core"))
            previous = mass
            continue
        area = np.diff(fine)*(f[:-1]+f[1:])/math.pi
        shell = (fine[:-1] < -extent+cfg["shell_width"]) | (fine[1:] > extent-cfg["shell_width"])
        edge_fraction = float(area[shell].sum()/mass)
        domain_change = abs(mass-previous)/mass if previous is not None else None
        row = dict(extent=extent, mass=mass, mesh=relative_mesh,
                   domain_change=domain_change, shell_fraction=edge_fraction,
                   sites=len(fine), status="continue")
        records.append(row)
        if (converged and domain_change is not None
                and domain_change <= cfg["mass_relative_tolerance"]
                and edge_fraction <= cfg["mass_relative_tolerance"]):
            cdf = np.r_[0., np.cumsum(area)]
            qlo, qhi = cfg["quantiles"]
            lo, hi = np.interp(np.array([qlo, qhi])*mass, cdf, fine)
            row["status"] = "resolved"
            return dict(t=fine, weights=w, mass=mass,
                        core=(float(lo)-cfg["guard"], float(hi)+cfg["guard"]),
                        history=records)
        previous = mass
    raise FitUnresolved("Raw core/mass survey unresolved: " + str(records))


class Objective:
    """Sampling and integration mesh stay fixed within each optimizer block."""
    def __init__(self, cfg, phase, work):
        self.cfg, self.phase, self.work = cfg, phase, work
        self.operator = Operators(phase["quadrature"])
        self.support = None

    def refresh(self, raw):
        self.support = survey(raw, self.cfg["support"], self.work)
        lo, hi = self.support["core"]
        self.t = np.linspace(lo, hi, self.phase["collocation"])
        self.w = torch.tensor(trapezoid_weights(self.t)/(hi-lo), dtype=torch.float64)

    def mass(self, raw):
        counted = CountedRaw(raw, self.work)
        x = torch.tensor(np.exp(self.support["t"]), dtype=torch.float64)
        return (counted(x)*torch.tensor(self.support["weights"], dtype=torch.float64)).sum()

    def __call__(self, raw):
        if self.support is None:
            raise ValueError("Refresh support before objective evaluation")
        model = CountedRaw(raw, self.work)
        mass = self.mass(raw)
        if not bool(torch.isfinite(mass)) or not bool(mass > 0):
            raise FitUnresolved("Nonpositive differentiable mass")
        x = torch.tensor(np.exp(self.t), dtype=torch.float64, requires_grad=True)
        h = model(x)
        D = x*torch.autograd.grad(h.sum(), x, create_graph=True)[0]
        g, V = self.operator(model, x)
        beta = raw.beta()
        lam = (1+beta)/(1-ANCHOR)
        terms = torch.stack((h, beta*D, ANCHOR*lam*((V/x)/mass)*D, -lam*(g/mass)*h))
        if not bool(torch.isfinite(terms).all()):
            raise FitUnresolved("Nonfinite fitting terms")
        scale = terms.detach().abs().amax(dim=0)
        if not bool((scale > 0).all()):
            raise FitUnresolved("Inactive fitting terms; no residual floor")
        scaled = terms/scale
        ratio = scaled.sum(0).square()/scaled.square().sum(0)
        return (self.w*ratio).sum()  # Physics scale 1; gauges are encoded.

    def validate(self, raw):
        """Fresh candidate-specific support and a doubled proxy, never a judge."""
        self.refresh(raw)
        coarse = float(self(raw).detach())
        fine_phase = dict(self.phase, quadrature=2*self.phase["quadrature"],
                          collocation=2*self.phase["collocation"]-1)
        fine = Objective(self.cfg, fine_phase, self.work)
        fine.support = self.support
        fine.t = np.linspace(*self.support["core"], fine_phase["collocation"])
        fine.w = torch.tensor(trapezoid_weights(fine.t)/(fine.t[-1]-fine.t[0]), dtype=torch.float64)
        value = float(fine(raw).detach())
        delta = abs(coarse-value)
        return dict(resolved=delta <= self.phase["proxy_absolute_tolerance"],
                    coarse=coarse, fine=value, absolute_change=delta,
                    tolerance=self.phase["proxy_absolute_tolerance"],
                    raw_core=list(self.support["core"]),
                    canonical_core=[v-math.log(self.support["mass"]) for v in self.support["core"]],
                    mass=self.support["mass"], support_history=self.support["history"])


def fit(raw, cfg, phase_name):
    """Last complete measured update; one declared fit, no restart/best-state pick."""
    phase = cfg[phase_name]
    work = Work(phase["seconds"], phase["max_field_elements"])
    objective = Objective(cfg, phase, work)
    committed = copy.deepcopy(raw.state_dict())
    trace, evaluations, updates, initial_loss = [], 0, 0, None
    stop, failure, validation = "stage_limits", None, None
    try:
        for spec in phase["stages"]:
            optimizer = None
            for update in range(spec["updates"]):
                if time.monotonic() >= work.deadline-phase["validation_reserve_seconds"]:
                    stop = "fit_time"
                    break
                if update % phase["refresh_updates"] == 0:
                    objective.refresh(raw)
                    # Adam moments persist; changed-mesh LBFGS history is reset.
                    if spec["kind"] == "lbfgs":
                        optimizer = None
                if optimizer is None:
                    if spec["kind"] == "adam":
                        optimizer = torch.optim.Adam(raw.parameters(), lr=spec["lr"])
                    elif spec["kind"] == "lbfgs":
                        optimizer = torch.optim.LBFGS(raw.parameters(), lr=spec["lr"],
                            max_iter=1, max_eval=20, history_size=20,
                            tolerance_grad=1e-10, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")
                    else:
                        raise ValueError("Unknown optimizer stage")
                if spec.get("cosine"):
                    lr = spec["lr"]*.5*(1+math.cos(math.pi*update/spec["updates"]))
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                def closure():
                    nonlocal evaluations, initial_loss
                    work.check()
                    if evaluations >= phase["max_closures"]:
                        raise TimeoutError("Fit closure ceiling")
                    raw.zero_grad(set_to_none=True)
                    value = objective(raw)
                    if not bool(torch.isfinite(value)):
                        raise FitUnresolved("Nonfinite fitting objective")
                    value.backward()
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in raw.parameters()):
                        raise FitUnresolved("Nonfinite fitting gradient")
                    evaluations += 1
                    if initial_loss is None:
                        initial_loss = float(value.detach())
                    return value
                if spec["kind"] == "adam":
                    closure(); optimizer.step()
                else:
                    optimizer.step(closure)
                observed = float(objective(raw).detach())
                if not math.isfinite(observed):
                    raise FitUnresolved("Unmeasured update")
                committed = copy.deepcopy(raw.state_dict())
                updates += 1
                trace.append(dict(update=updates, phase=spec["kind"], loss=observed,
                                  seconds=time.monotonic()-work.start))
            if stop != "stage_limits":
                break
    except TimeoutError as error:
        stop = str(error)
    except (ValueError, RuntimeError, OverflowError) as error:
        stop, failure = "failed", type(error).__name__+": "+str(error)
    raw.load_state_dict(committed)
    if failure is None:
        try:
            validation = objective.validate(raw)
        except (TimeoutError, ValueError, RuntimeError, OverflowError) as error:
            failure = type(error).__name__+": "+str(error)
    eligible = failure is None and updates > 0 and validation is not None and validation["resolved"]
    frozen = FrozenProfile(raw, validation["mass"]) if eligible else None
    return frozen, dict(eligible=eligible, stop=stop, failure=failure,
        initial_loss=initial_loss, updates=updates, closures=evaluations,
        selected_state="last_complete_measured_update", trace=trace,
        validation=validation, work=work.summary())
