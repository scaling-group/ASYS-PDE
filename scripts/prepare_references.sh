#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${ASYS_PYTHON:-python3}
case_name=${1:---help}
if [[ "$case_name" == --help ]]; then
    cat <<'HELP'
Usage: scripts/prepare_references.sh {fujita|nls|allen-cahn|allen-cahn-original|allen-cahn-geometric|gclm|--check} [OUTPUT_ROOT]
Generate numerical assessment references, separately from equation-based search.
Defaults to ./references (ignored by Git). Set ASYS_PYTHON to a scientific Python.
Full runs can take hours. NLS stores six full-resolution fields (several GB);
Allen-Cahn uses 512x512 Fourier grids; gCLM uses 4097 log-radius nodes.
--check runs small analytical fixtures only, without producing reference banks.
Completed outputs are verified by SHA256 before reuse. Incomplete outputs are
preserved; choose a fresh output root to retry them.
HELP
    exit 0
fi
case "$case_name" in fujita|nls|allen-cahn|allen-cahn-original|allen-cahn-geometric|gclm|--check) ;; *) echo "Unknown experiment: $case_name" >&2; exit 2 ;; esac
export PYTHONPATH="$root/scripts/references${PYTHONPATH:+:$PYTHONPATH}"
validate_outputs() {
    "$python" - "$case_name" "$target" <<'CHECK'
import json
import sys
from pathlib import Path

import numpy as np

case, directory = sys.argv[1:]
root = Path(directory)
if case == 'fujita':
    result = json.loads((root / 'fujita-forward.json').read_text())
    assert result['qualified']
    with np.load(root / 'fujita-forward.npz') as data:
        assert data['u'].shape == (289, 3549)
    solves = result['refinement_evidence']['solves']
    for method in ('finite_volume', 'nodal_finite_difference'):
        assert any(s['method'] == method and s['intervals'] == 16384
                   and s['rtol'] == 2e-11 and s['atol'] == 2e-13 for s in solves)
elif case == 'nls':
    for width in (3.5, 4., 4.5, 5., 6., 7.):
        folder = root / f'L{width:g}'
        result = json.loads((folder / 'reference_receipt.json').read_text())
        assert result['reference_refinement_passed']
        assert result['levels'] == [[4096, 512], [8192, 512], [4096, 1024]]
        with np.load(folder / 'reference.npz') as data:
            assert data['readout_state'].shape == (161, 1024)
            assert data['readout_times'].shape == (161,)
elif case in ('allen-cahn', 'allen-cahn-original'):
    for n, dt, count in ((384, 1/512, 262), (512, 1/1024, 12)):
        result = json.loads((root / f'etdrk4-n{n}.json').read_text())
        assert result['method'] == 'ETDRK4' and result['cells'] == n
        assert result['maximum_dt'] == dt and result['time_knots'] == count
        floor = 1e-14 if case == 'allen-cahn-original' else 1e-12
        assert result['initial_regularizer'] == floor and result['contour_points'] == 32
        assert (root / f'etdrk4-n{n}.npz').is_file()
elif case == 'allen-cahn-geometric':
    times = np.round(np.linspace(0, 8, 20)/.004)*.004
    for n, dt in ((256, .002), (256, .004), (512, .002)):
        with np.load(root / f'n{n}-dt{dt:g}.npz') as data:
            assert np.array_equal(data['t'], times)
CHECK
}
output=${2:-"$root/references"}
if [[ "$case_name" != --check ]]; then
    mkdir -p -- "$output"
    output=$(cd -- "$output" && pwd)
    target="$output/$case_name"
    if [[ -f "$target/SHA256SUMS" ]]; then
        (cd -- "$target"
            find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.tmp -print0 | sort -z | xargs -0 sha256sum | cmp - SHA256SUMS
        )
        validate_outputs
        echo "Verified existing $target"
        exit 0
    fi
    if [[ -e "$target" ]]; then
        echo "Incomplete output exists: $target; choose a fresh output root." >&2
        exit 1
    fi
    mkdir -- "$target"
fi
if [[ "$case_name" == fujita ]]; then
    "$python" "$root/scripts/references/fujita.py" --output "$target"
else
    "$python" - "$case_name" "${target:-}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

case, output = sys.argv[1:]
root = Path(output)
if case == '--check':
    import allen_cahn
    import fujita
    import nls
    from integral_reference import kernel

    allen_cahn.calibration(1)
    allen_cahn.calibration_etdrk4()
    r, operator = fujita.operator(64, 'nodal_finite_difference')
    assert np.allclose((operator @ (r*r))[:-1], 10., rtol=0, atol=1e-10)
    field, _ = fujita.solve(64, 'nodal_finite_difference', 1e-7, 1e-9,
                            np.linspace(0, 5, 21), np.array([0., 1e-6]))
    assert field.shape == (2, 21) and np.isfinite(field).all()
    nls.plane_wave()
    assert abs(kernel(20) - 2 * np.exp(-40) / 3) < 1e-30
    print('PASS: Fujita radial stencil/solve, Allen-Cahn exact flows and ETDRK4, NLS plane waves, gCLM kernel')
elif case == 'nls':
    import nls

    for width in (3.5, 4., 4.5, 5., 6., 7.):
        nls.generate(root / f'L{width:g}', width)
elif case in ('allen-cahn', 'allen-cahn-original'):
    import allen_cahn

    allen_cahn.calibration_etdrk4()
    main_times = np.unique(np.r_[0, 2.**np.arange(-10, -5), np.arange(1, 257)/32])
    fine_times = np.array([0, 1/1024, 1/256, 1/64, .05, .1, .5, 1, 2, 4, 6, 8])
    for n, dt, times in ((384, 1/512, main_times), (512, 1/1024, fine_times)):
        if case == 'allen-cahn-original':
            initial = lambda x, y: allen_cahn.initial_condition(x, y, 1e-14)
            x, values, diagnostics = allen_cahn.integrate_etdrk4(n, times, dt, initial)
            diagnostics['initial_regularizer'] = 1e-14
        else:
            x, values, diagnostics = allen_cahn.integrate_etdrk4(n, times, dt)
        np.savez_compressed(root / f'etdrk4-n{n}.npz', x=x, y=x, t=times, u=values)
        (root / f'etdrk4-n{n}.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
elif case == 'allen-cahn-geometric':
    import allen_cahn

    allen_cahn.calibration(1)
    times = np.round(np.linspace(0, 8, 20)/.004)*.004
    for n, dt in ((256, .002), (256, .004), (512, .002)):
        x, values, diagnostics = allen_cahn.integrate(n, times, dt, 1)
        np.savez_compressed(root / f'n{n}-dt{dt:g}.npz', x=x, y=x, t=times, u=values)
        (root / f'n{n}-dt{dt:g}.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
elif case == 'gclm':
    from fixed_point_reference import LogProfile, iterate

    config = dict(integration_extent=40., integration_tolerance=1e-11,
                  panel_width=2., singularity_rule='weighted_log',
                  maximum_iterations=120, maximum_seconds=1800,
                  iteration_tolerance=1e-11)
    for label, size, extent in (('coarse', 1025, 18), ('medium', 2049, 18),
                                ('fine', 4097, 18), ('extended_domain', 4097, 20)):
        w = np.linspace(-extent, extent, size)
        seed = LogProfile(w, -np.log1p(np.exp(2*w)), 2.)
        profile, result = iterate(seed, .25, config)
        if not result['converged']:
            raise RuntimeError(f'gCLM {label} iteration did not converge')
        constants = result['constants']
        destination = root / label
        destination.mkdir()
        np.savez(destination / 'profile.npz', w=profile.w, log_values=profile.log_values,
                 tail_power=profile.tail_power, b=constants['b'], beta=constants['beta'], a=.25)
        (destination / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
PY
fi
if [[ "$case_name" != --check ]]; then
    validate_outputs
    (cd -- "$target"
        find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.tmp -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.tmp
        mv -- SHA256SUMS.tmp SHA256SUMS
    )
    echo "Prepared $target"
fi
