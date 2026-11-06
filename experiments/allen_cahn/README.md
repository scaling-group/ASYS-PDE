# Coupled Allen–Cahn interface modes

`selected/formula14/` contains the manuscript's selected 14-scalar expression;
`selected/formula12/` contains its 12-scalar extension. Each directory supplies
the original model, configuration, coefficient state and fitting definitions.
`neural/` supplies the current 12,801-parameter control, its equation-based
fitter and all three selected states.

The neural model uses a four-layer, width-64 tanh network to correct a prescribed
two-interface field. It preserves the initial trace and tangent, a bounded
output and periodic outer closure. Its coefficients are selected by the lowest
finite training loss measured after a completed L-BFGS step. Incomplete steps
are rolled back; numerical reference fields do not enter fitting or selection.

Two reference conventions are retained. The original-formula references use a squared
radial regularizer of `1e-14`; the current neural comparison uses `1e-12`.
Both use epsilon `0.04`. Their saved trajectory errors are:

| Construction | Relative field error |
| --- | ---: |
| 14-scalar formula | 0.7981% |
| 12-scalar formula | 0.9162% |
| Neural seed 2026110401 | 0.1224% |
| Neural seed 2026110411 | 0.1148% |
| Neural seed 2026110421 | 0.1384% |

Generate and assess the two banks from the repository root:

```bash
scripts/prepare_references.sh allen-cahn-original
scripts/prepare_references.sh allen-cahn
uv run python experiments/allen_cahn/evaluate.py \
  --original-reference references/allen-cahn-original/etdrk4-n384.npz \
  --current-reference references/allen-cahn/etdrk4-n384.npz \
  --output /tmp/allen-cahn.json
```

Either reference argument can be used alone. `--original-reference` assesses
both selected formulas against the `1e-14` bank. `--current-reference` assesses
the same frozen 14-scalar formula and all three neural controls against the
`1e-12` bank, providing their comparison on a common reference. The readout
integrates squared field errors and reference energies over all 262 saved times
on the 384-square periodic grid, then takes their ratio's square root. It also reports the
reference interface-band error and symmetric periodic contour distances.
`expected.json` gives the saved results with their reference labels.

To refit a neural control, use a fresh output directory:

```bash
uv run python experiments/allen_cahn/neural/shared_runner.py \
  --arm nn-lbfgs --seed 2026110401 --out /tmp/ac-neural-refit
```

The 1,200-second process allowance includes setup and saving, with 15 seconds
reserved for finalization. Each run uses 4,096 residual, 512 initial and 64
boundary samples, plus a focused residual on the same spatial points scaled by
`0.55`. The loss and optimizer settings are preserved in the fitter. The older
geometric study's hard-jet control remains under `baseline/allen_cahn/`.
