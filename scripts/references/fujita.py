"""Refined radial Fujita IVP solves with mesh, method and tolerance checks."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.sparse import diags

R, T_STAR = 5.0, 0.020406


def initial(r):
    return 5 * np.cos(np.pi * r / 10)


def operator(n, method):
    h = R / n
    if method == "finite_volume":
        faces = np.linspace(0, R, n + 1)
        r = (faces[:-1] + faces[1:]) / 2
        volume = (faces[1:] ** 5 - faces[:-1] ** 5) / 5
        conductance = faces[1:-1] ** 4 / h
        lower, upper = conductance / volume[1:], conductance / volume[:-1]
        diagonal = np.zeros(n)
        diagonal[:-1] -= upper
        diagonal[1:] -= lower
        diagonal[-1] -= 2 * R**4 / (h * volume[-1])
    else:
        r = np.arange(n) * h
        lower = (1 - 2 / np.arange(1, n)) / h**2
        upper = np.r_[10.0, 1 + 2 / np.arange(1, n - 1)] / h**2
        diagonal = np.r_[-10.0, np.full(n - 1, -2.0)] / h**2
    return r, diags((lower, diagonal, upper), (-1, 0, 1), format="csc")


def solve(n, method, rtol, atol, assessment_r, assessment_t):
    r, linear = operator(n, method)

    def rhs(t, u):
        return linear @ u + u**3

    def jac(t, u):
        return linear + diags(3 * u * u, format="csc")

    def overflow(t, u):
        return float(u.max() - 1000)

    overflow.terminal, overflow.direction = True, 1
    before, cpu_before = time.monotonic(), time.process_time()
    answer = solve_ivp(
        rhs,
        (0, assessment_t[-1]),
        initial(r),
        method="BDF",
        jac=jac,
        rtol=rtol,
        atol=atol,
        t_eval=assessment_t,
        events=overflow,
    )
    if not answer.success or len(answer.t) != len(assessment_t):
        raise ValueError(f"IVP incomplete: {method}, n={n}: {answer.message}; end={answer.t[-1]}")
    if method == "finite_volume":
        grid = np.r_[-R, -r[::-1], r, R]
        values = np.vstack(
            (np.zeros(len(answer.t)), answer.y[::-1], answer.y, np.zeros(len(answer.t)))
        )
        spline = CubicSpline(grid, values, axis=0)
    else:
        grid = np.r_[r, R]
        values = np.vstack((answer.y, np.zeros(len(answer.t))))
        spline = CubicSpline(
            grid, values, axis=0, bc_type=((1, np.zeros(len(answer.t))), "not-a-knot")
        )
    field = spline(assessment_r).T
    field[0] = initial(assessment_r)
    field[:, -1] = 0
    receipt = dict(
        method=method,
        intervals=n,
        dr=R / n,
        integrator="BDF",
        rtol=rtol,
        atol=atol,
        nfev=answer.nfev,
        njev=answer.njev,
        nlu=answer.nlu,
        elapsed_seconds=time.monotonic() - before,
        cpu_seconds=time.process_time() - cpu_before,
        minimum=float(field.min()),
        final_peak=float(field[-1].max()),
        reconstruction="even cubic interpolation for FV; origin derivative zero for nodal FD",
        initialization="analytic midpoint values for FV; analytic nodal values for FD",
        exact_slices="analytic t=0; boundary value zero at r=5",
    )
    return field, receipt


def weights(x):
    return np.r_[(x[1] - x[0]) / 2, (x[2:] - x[:-2]) / 2, (x[-1] - x[-2]) / 2]


def relative(a, b, w):
    return float(np.sqrt(np.sum(w * (a - b) ** 2) / np.sum(w * b * b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    DESTINATION = parser.parse_args().output
    started, cpu_started = time.monotonic(), time.process_time()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    generator = Path(__file__)
    generator_sha = hashlib.sha256(generator.read_bytes()).hexdigest()
    r = np.unique(np.r_[np.linspace(0, R, 2049), np.linspace(0, 0.15, 1501)])
    t = np.unique(
        np.r_[np.linspace(0, 0.95 * T_STAR, 193), T_STAR * (1 - np.geomspace(0.05, 0.001, 97))]
    )
    w = weights(t)[:, None] * weights(r)[None, :]
    records, previous = [], None
    for method, n, rtol, atol in (
        ("finite_volume", 1024, 2e-9, 2e-11),
        ("finite_volume", 2048, 2e-9, 2e-11),
        ("finite_volume", 4096, 2e-9, 2e-11),
        ("finite_volume", 8192, 2e-9, 2e-11),
        ("nodal_finite_difference", 4096, 2e-9, 2e-11),
        ("nodal_finite_difference", 8192, 2e-9, 2e-11),
        ("finite_volume", 8192, 2e-10, 2e-12),
    ):
        field, receipt = solve(n, method, rtol, atol, r, t)
        if previous is not None:
            receipt["relative_l2_to_previous"] = relative(field, previous, w)
            receipt["final_peak_relative_change_to_previous"] = (
                abs(field[-1, 0] - previous[-1, 0]) / field[-1, 0]
            )
        if method == "finite_volume" and n == 4096:
            fv4096 = field.copy()
        if method == "finite_volume" and n == 8192 and rtol == 2e-9:
            fv8192 = field.copy()
        if method == "nodal_finite_difference" and n == 8192:
            fd8192 = field.copy()
        records.append(receipt)
        previous = field
        print(json.dumps(receipt), flush=True)
    refinement = dict(
        fv_4096_8192_relative_l2=relative(fv4096, fv8192, w),
        independent_fv_fd_8192_relative_l2=relative(fv8192, fd8192, w),
        bdf_tolerance_relative_l2=relative(fv8192, field, w),
        fv_4096_8192_final_peak_relative_change=abs(fv4096[-1, 0] - fv8192[-1, 0]) / fv8192[-1, 0],
        independent_final_peak_relative_change=abs(fd8192[-1, 0] - field[-1, 0]) / field[-1, 0],
    )
    fv16384, fv_receipt = solve(16384, "finite_volume", 2e-11, 2e-13, r, t)
    fd16384, fd_receipt = solve(16384, "nodal_finite_difference", 2e-11, 2e-13, r, t)
    records.extend((fv_receipt, fd_receipt))
    refinement.update(
        fv_8192_16384_tight_relative_l2=relative(field, fv16384, w),
        independent_fv_fd_16384_relative_l2=relative(fd16384, fv16384, w),
        independent_fv_fd_16384_final_relative_l2=relative(fd16384[-1], fv16384[-1], weights(r)),
    )
    checks = dict(
        finite=bool(all(np.isfinite(a).all() for a in (field, fv16384, fd16384))),
        finer_mesh_l2_below_1e_3=bool(refinement["fv_8192_16384_tight_relative_l2"] < 1e-3),
        finer_independent_methods_l2_below_1e_3=bool(
            refinement["independent_fv_fd_16384_relative_l2"] < 1e-3
        ),
        nonnegative=bool(field.min() >= -1e-8),
        horizon_precedes_supplied_t_star=bool(t[-1] < T_STAR),
        fv_mesh_l2_below_1e_3=bool(refinement["fv_4096_8192_relative_l2"] < 1e-3),
        independent_methods_l2_below_1e_3=bool(
            refinement["independent_fv_fd_8192_relative_l2"] < 1e-3
        ),
        bdf_tolerance_l2_below_1e_5=bool(refinement["bdf_tolerance_relative_l2"] < 1e-5),
        final_peak_mesh_and_method_change_below_5e_3=bool(
            max(
                refinement["fv_4096_8192_final_peak_relative_change"],
                refinement["independent_final_peak_relative_change"],
            )
            < 5e-3
        ),
    )
    path = DESTINATION / "fujita-forward.npz"
    np.savez_compressed(path, r=r, t=t, u=field, t_star=np.array(T_STAR))
    manifest = dict(
        qualified=all(checks.values()),
        case="fujita_trajectory",
        file=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        metric="resolved_trapezoid",
        field_layout="time_space",
        generator_provenance=dict(
            file=generator.name,
            sha256=generator_sha,
            numpy=np.__version__,
            scipy=scipy.__version__,
            equation=(
                "u_t=u_rr+4*u_r/r+u^3; r in [0,5]; u0=5*cos(pi*r/10); "
                "Neumann origin; Dirichlet at 5"
            ),
        ),
        refinement_evidence=dict(solves=records, comparisons=refinement, checks=checks),
        similarity_metadata=dict(
            t_star=T_STAR,
            role="sampling clock",
            horizon=0.999 * T_STAR,
        ),
        grid=dict(
            r_points=len(r),
            t_points=len(t),
            r_min=0,
            r_max=5,
            t_min=0,
            t_max=float(t[-1]),
            core_spacing=0.0001,
        ),
        norm="trapezoidal dr dt; no radial r^4 weighting",
        elapsed_seconds=time.monotonic() - started,
        cpu_seconds=time.process_time() - cpu_started,
    )
    (DESTINATION / "fujita-forward.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(dict(qualified=manifest["qualified"], checks=checks, comparisons=refinement)),
        flush=True,
    )

    if not manifest["qualified"]:
        raise RuntimeError("Fujita reference refinement checks failed")


if __name__ == "__main__":
    main()
