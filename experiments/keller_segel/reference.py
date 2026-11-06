"""Cumulative-mass BDF reference on the original 77-time schedule."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

def mass_operator(radius):
    # Exact algebra from the retained compare_be6d.py, with n=96000.
    r = radius[1:-1]
    h = radius[1] - radius[0]

    def rhs(t, y):
        m = np.r_[0., y, 4.75]
        return ((m[2:] - 2*y + m[:-2]) / h**2
                + (y - 1) * (m[2:] - m[:-2]) / (2*h*r))

    def jac(t, y):
        m = np.r_[0., y, 4.75]
        lower = 1/h**2 - (y - 1)/(2*h*r)
        upper = 1/h**2 + (y - 1)/(2*h*r)
        center = -2/h**2 + (m[2:] - m[:-2])/(2*h*r)
        return diags([lower[1:], center, upper[:-1]], [-1, 0, 1], format="csc")

    return rhs, jac


def check_jacobian():
    # A small complex-step algebra check; no additional IVP integration.
    radius = np.linspace(0., 30., 42)
    y = -4.75*np.expm1(-radius[1:-1]**2)
    direction = np.sin(radius[1:-1])
    rhs, jac = mass_operator(radius)
    numeric = rhs(0., y.astype(complex) + 1e-30j*direction).imag/1e-30
    analytic = jac(0., y) @ direction
    error = float(np.max(np.abs(numeric-analytic))/(1+np.max(np.abs(numeric))))
    if error > 1e-10:
        raise ValueError("Retained mass Jacobian algebra check failed")
    return error


def density_nodes(radius, mass, stride):
    # Unchanged reference_readout.py reconstruction, before native-node selection.
    h = radius[1] - radius[0]
    rows = []
    for i in range(mass.shape[1]):
        m = np.r_[0., mass[:, i], 4.75]
        density = np.r_[2*m[1]/h**2,
                        (m[2:] - m[:-2])/(2*h*radius[1:-1]), 0.]
        rows.append(density[::stride])
    return np.asarray(rows)


def generate(output, cells):
    if output.exists():
        raise FileExistsError(output)
    check_jacobian()
    times = np.unique(np.r_[np.linspace(0., .86652, 65),
                           [.0005,.001,.002,.005,.01,.02,.05,.1,.2,.4,.6,.8]])
    radius = np.linspace(0., 30., cells+1)
    rhs, jac = mass_operator(radius)
    solution = solve_ivp(rhs, (0., .86652), -4.75*np.expm1(-radius[1:-1]**2),
                         method='BDF', jac=jac, t_eval=times, rtol=2e-9,
                         atol=2e-11, max_step=.01)
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError(solution.message)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, radius=radius, times=times, mass=solution.y,
                        rtol=2e-9, atol=2e-11, max_step=.01)
    output.with_suffix('.json').write_text(json.dumps(dict(cells=cells, times=77,
        equation='M_t=M_rr+(M-1)*M_r/r', method='BDF', rtol=2e-9, atol=2e-11,
        max_step=.01, nfev=solution.nfev, njev=solution.njev), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, choices=(96000,192000), default=192000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    generate(args.output, args.cells)

