# Periodic NLS and width transfer

The manuscript assesses three prescribed fields: the line core with periodic
boundary radiation (`C+B`), its first nonlinear-defect response (`C+B+P`), and
the next response (`C+B+P+E`). They solve the same representation problem for
the periodic extension of `2 sech(x)` on boxes of half-width `L`.

`model.py` and `config.json` reconstruct those fields with predictor orders
`0`, `1` and `2`. All eight learned deviations remain zero. The constructor
builds the finite analytic quadrature and Gram-conditioned response basis; no
fitted checkpoint or numerical PDE trajectory is needed to construct them.
The supplied helpers preserve the original core, weak periodic compatibility
and response definitions.

From the repository root:

```bash
scripts/prepare_references.sh nls
uv run python experiments/nonlinear_schrodinger/evaluate.py --reference-root references/nls --output /tmp/nls.json
```

The readout uses 1,024 periodic spatial points and 161 stored times on
`[0, pi/2]`, with uniform spatial means and trapezoidal time weights. It reports
full-field error, error relative to the reference's correction to the line
breather, and mass drift. No phase alignment is applied. `expected.json` contains
the saved comparisons and spatial/time reference refinements; its errors are
fractions. Widths `4, 5, 6` were used in construction, and `3.5, 4.5, 7` test
transfer. The pooled result sums error and reference energies over the requested
widths before taking their ratio's square root, with no extra box-length weight.
Across all six widths, the full-field error is approximately `0.530%` for
`C+B` and `0.0250%` for `C+B+P+E`. Use `--widths` and `--orders`
to assess a subset; the output identifies the widths included in each pooled row.

The refined mass comparison uses 4,096 spatial points for `L <= 4.5` and
2,048 for wider boxes, at 321 uniform times. It needs no reference bank:

```bash
uv run python experiments/nonlinear_schrodinger/mass.py --output /tmp/nls-mass.json
```

For temporal sensitivity in the narrowest boxes, add
`--widths 3.5 4 --time-counts 321 641 1281`. At `L=3.5`, the saved predictor and
defect-corrected mass drifts are approximately `2.52` and `0.0777` parts per million.
