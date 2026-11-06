"""Radial-profile references: collocation plus independent two-sided IVP."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import root

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
DESTINATION = parser.parse_args().output
ANCHOR = 'f5bd9987fada497e789969dde5657d070e7a2eee'
S = np.array([[0., 0.], [0., -4.]])
started = time.monotonic()


def ode(x, y, p=None):
    return np.vstack((y[1], .5*x*y[1] + .5*y[0] - y[0]**3))


def ivp(x, y):
    return [y[1], (.5*x-4/x)*y[1] + .5*y[0] - y[0]**3]


def tail(c, x):
    d = 2*c-c**3
    e = -1.5*c*c*d
    f = -(10+3*c*c)*e/3-c*d*d
    g = -((28+3*c*c)*f+6*c*d*e+d**3)/4
    powers = np.array([1, 3, 5, 7, 9])
    coefficients = np.array([c, d, e, f, g])
    return (np.sum(coefficients/x**powers),
            np.sum(-powers*coefficients/x**(powers+1)))


def collocation(previous, R, tol, guess_a, guess_b):
    x = np.unique(np.r_[np.linspace(0, 6, 601), np.geomspace(6, R, 401)])
    if previous is None:
        y = np.array([guess_a/np.sqrt(1+guess_b*x*x),
                      -guess_a*guess_b*x/(1+guess_b*x*x)**1.5])
        c = float(R*y[0, -1])
    else:
        y = previous.sol(np.minimum(x, previous.x[-1]))
        c = float(previous.p[0])
        for i in np.flatnonzero(x > previous.x[-1]):
            y[:, i] = tail(c, x[i])
    def boundary(left, right, p):
        value, derivative = tail(p[0], R)
        return np.array([left[1], right[0]-value, right[1]-derivative])
    answer = solve_bvp(ode, boundary, x, y, p=np.array([c]), S=S,
                       tol=tol, bc_tol=tol/10, max_nodes=40000)
    if not answer.success:
        raise ValueError(f'collocation failed: R={R}, tol={tol}: {answer.message}')
    return answer


def weights(x):
    return np.r_[(x[1]-x[0])/2, (x[2:]-x[:-2])/2, (x[-1]-x[-2])/2]


def relative(a, b, x):
    return float(np.sqrt(np.sum(weights(x)*(a-b)**2)/np.sum(weights(x)*b*b)))


def shooting(a, c, R):
    epsilon, match = 1e-6, 3.
    def halves(log_parameters, dense=False):
        origin, coefficient = np.exp(log_parameters)
        b = (.5*origin-origin**3)/10
        d = (1.5-3*origin*origin)*b/28
        initial = [origin+b*epsilon**2+d*epsilon**4,
                   2*b*epsilon+4*d*epsilon**3]
        outward = solve_ivp(ivp, (epsilon, match), initial, method='DOP853',
                             rtol=2e-11, atol=2e-13, dense_output=dense)
        inward = solve_ivp(ivp, (R, match), tail(coefficient, R), method='Radau',
                            rtol=2e-11, atol=2e-13, dense_output=dense)
        if not (outward.success and inward.success):
            raise ValueError('independent two-sided integration failed')
        return outward, inward
    def mismatch(log_parameters):
        outward, inward = halves(log_parameters)
        return outward.y[:, -1]-inward.y[:, -1]
    answer = root(mismatch, np.log([a, c]), tol=1e-9)
    error = float(np.max(np.abs(mismatch(answer.x))))
    if error > 1e-9:
        raise ValueError(f'two-sided matching did not converge: {error}')
    origin, coefficient = np.exp(answer.x)
    outward, inward = halves(answer.x, True)
    def profile(x):
        result = np.empty_like(x)
        result[x == 0] = origin
        core = (x > 0) & (x <= match)
        middle = (x > match) & (x <= R)
        result[core] = outward.sol(x[core])[0]
        result[middle] = inward.sol(x[middle])[0]
        for i in np.flatnonzero(x > R): result[i] = tail(coefficient, x[i])[0]
        return result
    return profile, dict(R=R, method='DOP853 outward; Radau inward; root matching at eta=3',
                         rtol=2e-11, atol=2e-13, origin_epsilon=epsilon,
                         matching_error=error, a=float(origin), c=float(coefficient))


DESTINATION.mkdir(parents=True, exist_ok=False)
generator = Path(__file__)
generator_sha = hashlib.sha256(generator.read_bytes()).hexdigest()
eta = np.unique(np.r_[np.linspace(0, 10, 10001), np.geomspace(10, 350, 4001)])
records = []
for number, (guess_a, guess_b) in enumerate(((2., 1.), (10., 10.)), 1):
    solution = None
    refinements = []
    previous_values = None
    for R, tolerance in ((20., 1e-6), (40., 1e-7), (80., 1e-8),
                         (160., 1e-8), (350., 1e-8), (700., 1e-10)):
        solution = collocation(solution, R, tolerance, guess_a, guess_b)
        values = solution.sol(np.minimum(eta, R))[0]
        overlap = eta <= min(R, refinements[-1]['R']) if refinements else eta <= R
        change = relative(values[overlap], previous_values[overlap], eta[overlap]) if previous_values is not None else None
        refinements.append(dict(R=R, tolerance=tolerance, nodes=len(solution.x),
                                max_rms_residual=float(solution.rms_residuals.max()),
                                a=float(solution.y[0, 0]), c=float(solution.p[0]),
                                overlap_relative_l2_change=change))
        previous_values = values
    phi = solution.sol(eta)[0]
    independent = []
    for R in (40., 80.):
        profile, receipt = shooting(float(phi[0]), float(solution.p[0]), R)
        receipt['relative_l2_to_collocation'] = relative(profile(eta), phi, eta)
        receipt['max_absolute_to_collocation'] = float(np.max(np.abs(profile(eta)-phi)))
        independent.append(receipt)
    checks = dict(positive_on_dense_grid=bool(np.all(phi > 0)),
                  nonconstant=bool(phi[0] > np.sqrt(.5)+1e-3),
                  nonincreasing_on_dense_grid=bool(np.all(solution.sol(eta)[1] <= 1e-10)),
                  collocation_rms_below_2e_10=bool(solution.rms_residuals.max() < 2e-10),
                  domain_and_tolerance_change_below_1e_7=bool(refinements[-1]['overlap_relative_l2_change'] < 1e-7),
                  independent_methods_l2_below_1e_7=all(v['relative_l2_to_collocation'] < 1e-7 for v in independent),
                  independent_origins_relative_change_below_1e_7=bool(abs(independent[0]['a']-independent[1]['a'])/phi[0] < 1e-7))
    branch = f'observed-positive-branch-{number}'
    path = DESTINATION/f'{branch}.npz'
    np.savez_compressed(path, eta=eta, phi=phi)
    manifest = dict(qualified=all(checks.values()), case='fujita_profile', branch_id=branch,
                    branch_completeness='not claimed; two observed branches only',
                    file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    metric='resolved_trapezoid', generator_provenance=dict(
                        file=generator.name, sha256=generator_sha, numpy=np.__version__, scipy=scipy.__version__,
                        source_anchor=ANCHOR, equation='Phi"+4 Phi\'/eta+Phi^3-Phi/2-eta Phi\'/2=0',
                        initial_guesses='exploratory numerical basins, outside all fresh candidate inputs'),
                    refinement_evidence=dict(collocation=refinements, independent_shooting=independent, checks=checks),
                    grid=dict(eta_min=0, eta_max=350, core_step=.001, tail_log_points=4001, points=len(eta)),
                    boundary=dict(origin='regular radial expansion and derivative zero',
                                  far_field='algebraic expansion through eta^-9; domain through 700'))
    (DESTINATION/f'{branch}.json').write_text(json.dumps(manifest, indent=2)+'\n')
    records.append(manifest)
    print(json.dumps(dict(branch=branch, qualified=manifest['qualified'], checks=checks)), flush=True)
receipt = dict(model_calls=0, numerical_reference_work=True, generator_sha256=generator_sha,
               destination=str(DESTINATION), elapsed_seconds=time.monotonic()-started, records=records)
(DESTINATION/'qualification.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(dict(elapsed_seconds=receipt['elapsed_seconds'], all_qualified=all(r['qualified'] for r in records))), flush=True)
