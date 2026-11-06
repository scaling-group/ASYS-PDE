# Conditional logarithmic waiting profile

The 18-coefficient construction extends the supplied origin expansion of
`F F'' + (F')² - (7/3)x F' + (11/3)F = 0` on `0 <= x <= 0.01`.
Its exponential representation preserves positivity and the prescribed lower
jets, while retaining the forced, unsmoothed `x⁸ log(x)` term. The free resonant
coefficient and a 17-term Chebyshev remainder are restored from the original
selected state.

```bash
uv run python experiments/waiting_profile/evaluate.py --output /tmp/waiting-profile.json
```

The readout uses 257 logarithmic points from `1e-5` to `0.01` and normalizes the
ODE defect by the sum of the magnitudes of its four terms. The saved RMS is
`6.60e-17`, at float64 cancellation level; it is an equation defect, not a
profile-error estimate. The formula is C⁷ at the origin and has a divergent
eighth derivative. It does not establish a radial IVP solution or waiting-time
persistence.

`derivation/` contains the exact recurrence and a high-precision check of the
first unrepresented logarithmic terms. Run them with the baseline dependencies:

```bash
uv run --extra baseline python experiments/waiting_profile/derivation/polyhomogeneous.py
uv run --extra baseline python experiments/waiting_profile/derivation/asymptotic_audit.py --output /tmp/waiting-asymptotics.json
```
