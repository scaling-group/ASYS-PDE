# Stationary radial Fujita profile

The 38-scalar expression joins a regular positive core to a `1/eta` tail using
a degree-36 Chebyshev correction in a compactified coordinate. Its origin
curvature relation holds for every finite choice of coefficients.

`model.py` and `state_dict.pt` are the original selected source and fitted state.
`coefficients.json` also gives the manuscript's 17-significant-digit table in
the model's two parameter arrays. `source.json` records their provenance.

```bash
uv run python experiments/fujita_stationary/evaluate.py --output /tmp/fujita-profile.json
```

The formula gives origin value `6.63471855955`, origin second derivative
`-57.7479140847`, and leading tail coefficient `1.05689553114`. Assessment
reports the ODE defect separately from the finite-endpoint Robin defect.
Derivatives are computed independently from `forward` on the original
11,001-node residual grid: 10,001 equally spaced nodes on `[0,10]` joined to
1,001 geometric nodes on `[10,350]`. The peak interior ODE defect is about
`4.94e-12`; origin compatibility is checked using the regular radial limit.
Batch and train/eval invariance are checked separately. The reference profile
comparison below uses its own 14,001-node grid.

For an independent reference comparison:

```bash
uv run python experiments/fujita_stationary/reference.py --output references/fujita-stationary
uv run python experiments/fujita_stationary/evaluate.py \
  --reference references/fujita-stationary/observed-positive-branch-1.npz \
  --output /tmp/fujita-profile-reference.json
```

The reference generator retains the original collocation continuation and
independent two-sided integration checks. It finds two observed positive
branches; the manuscript uses branch 1. The saved relative profile error is
`2.70e-9%`, comparable to the reference methods' numerical difference.
