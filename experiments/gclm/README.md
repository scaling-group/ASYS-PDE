# Real-line gCLM profile and tail

The returned cohort comprises five ASYS constructions and four equation-selected
SIGS/FEX controls at `a=0.25`. The selected H3 construction combines polynomial, fractional-power and rational
shape directions in a compactified coordinate. Its 14 shape coefficients and
fitted exponent describe the normalized profile at `a = 0.25`.

`selected/` contains the original model, configuration, frozen normalization and
state used for the manuscript's H3 profile. Its relative error on 257 logarithmic
radii from `0.001` to `1000` is approximately `2.47e-8%` against the fine integral
reference. Normalization is restored from the saved state; it is not refitted
when assessing the profile.

`cohort/` contains the other four original ASYS models and states, plus the two
selected SIGS and two selected FEX outcomes supplied with the fractional
coordinate. H3 reuses `selected/`. The manifest identifies the selected revision
or terminal for each outcome. Their original coefficients and normalization
are restored directly; evaluating a profile does not recompute its gauge.
The original mathematical source is preserved, and shared reconstruction and
assessment helpers are reused from the release.

`matched_coordinates/` contains the separate four-way comparison: polynomial
and ASYS shape families, each fitted in identity or Gram-conditioned coordinates.
All four hold the exponent at `0.6845136212067134`, start with zero shape
coefficients and complete 20,000 Adam updates. The ASYS Gram result has profile
error `2.23e-8%`; it is a separate fit from H3. `expected.json` records the profile
and analytic two-term tail-balance results for all four arms.

From the repository root:

```bash
scripts/prepare_references.sh gclm
uv run --extra baseline python experiments/gclm/evaluate.py --reference-root references/gclm --output /tmp/gclm-h3.json
uv run python experiments/gclm/evaluate.py --cohort --reference-root references/gclm --output /tmp/gclm-cohort.json
uv run --extra baseline python experiments/gclm/evaluate.py --matched --reference-root references/gclm --output /tmp/gclm-matched.json
```

The cohort command reports all nine profiles against the coarse, medium, fine
and extended-domain banks, and aggregates equal-weight search-outcome medians.
The primary error is the unweighted relative L2 error on 257 logarithmic radii
from `0.001` to `1000`; physical-radius weighted L2, peak-normalized maximum
error and absolute exponent error are also reported. Profile errors in the
command output are percentages. The original `cohort/expected.json` retains
fraction-valued errors and sampled profiles for independent comparison.

| Method | Outcomes | Fine-bank median profile error (%) |
| --- | ---: | ---: |
| ASYS | 5 | `3.8960768e-5` |
| SIGS | 2 | `0.011917036` |
| FEX | 2 | `0.008509830` |

These are the manuscript's `3.90e-5%`, `0.0119%` and `0.00851%` medians.
Add `--public` to the cohort command to rerun the fixed equation assessor as
well. This evaluates the frozen fields and can take longer; it does not
reselect candidates. `cohort/source.json` records the original artifact hashes,
and the manifest records reference-bank hashes and reused sources.

The H3 and matched tail calculations use the same 80-digit endpoint algebra
through `mpmath` (included in the `baseline` extra). The H3 command returns
`C=1.7537812473`, `D=1.2678023973` and signed balance error `0.0024930385%`,
reproducing the manuscript's approximately `0.0025%` rate agreement.
`selected/expected.json` binds the original published tail coefficients to
this check. The readout also gives `lambda_D`, the amplitude of the limiting
physical-time derivative, and the PDE coefficient `K` in `D=C^2 K`.
The calculation does not estimate tail coefficients by subtracting nearly equal far-field
values. The reference banks are used only after the candidate states are fixed.

To refit any returned ASYS family from its original initialization, choose
`H1` through `H5` and a fresh output directory:

```bash
uv run python experiments/gclm/refit.py --history H3 --output /tmp/gclm-h3-refit
```

This uses the shared support-aware fitter, the original 20,000-update terminal
schedule with its 1,200-second cap, and the recorded initialization seed.
The emitted state includes newly frozen normalization; the released state is
unchanged. New stochastic searches use
`configs/eve/application/asys-gclm-profile`. The SIGS/FEX search commands and
fitting policies are in [baseline/gclm](../../baseline/gclm/README.md); their
`tail_capable` cohort corresponds to the four returned controls here.

To refit a matched arm, choose a fresh output directory:

```bash
ASYS_SEED=2 MATCHED_ARM=asys_gram uv run python experiments/gclm/matched_coordinates/source/asys_eval.py --log /tmp/gclm-refit
```

The other arm names are `poly_identity`, `poly_gram` and `asys_identity`.
The source includes the support-aware fitting objective and fixed assessment
settings. The fit retains the last completed update, checks its support and
freezes its normalization. These numerical controls do not launch agent searches.
