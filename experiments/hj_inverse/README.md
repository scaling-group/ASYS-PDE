# Hamilton–Jacobi inference with exponential time functions

`polynomial/` and `exponential/` contain the original paired field definitions,
configurations, fitting recipes and selected checkpoints. Both use exact heat
evolution plus 7,542 Fourier correction amplitudes and fit four independent
physical coefficients. The second replaces the six polynomial time directions
by six directions spanned by `1-exp(-m t/(2H))`, `m=1,...,6`, in conditioned
coordinates. Neither assumes a common Cole–Hopf ratio.

Both controls start with zero correction amplitudes and use 20,000 Adam updates
with a cosine learning rate from `0.003` to `0.00006`. `problem.json` specifies
the supplied initial modes, coefficient starting values and fitting horizon.
The observation packet and fixed task helpers are shipped with
`configs/eve/application/asys-hj-inverse/`. The packet's reference coefficients
are `(d_x,d_y,b_x,b_y)=(0.115,0.085,-0.43,-0.61)`.

Replay both selected constructions from the repository root:

```bash
uv run python experiments/hj_inverse/evaluate.py --output /tmp/hj-inverse.json
```

The evaluator verifies each model, problem and checkpoint hash, restores the
spectral amplitudes and the four physical coefficients, and evaluates the
expressions without fitting. The fitted diffusivities enter the exact heat
component. Each `selected-state.pt` is the original full evaluator state;
`expected.json` records its identity and the original measurements.

The field metric is a relative Frobenius norm on 64-square spatial grids and
50 equally weighted frames from `0` to `0.5`. Frames 0–34 cover fitting and
35–49 assess the future. Coefficient error is the mean of the four relative
absolute errors. The saved results are:

| Time basis | Future-field error | Final-time field error | Coefficient error |
| --- | ---: | ---: | ---: |
| Polynomial | 3.3572% | 8.2663% | 0.7818% |
| Exponential | 1.0645% | 2.3666% | 0.7842% |

The JSON output also reports every frame and the pooled fitting/full-window
errors; its relative errors are fractions. These measurements assess the
returned fields directly. A PDE rollout using inferred coefficients is a
separate endpoint, and this comparison does not establish coefficient
identifiability.

`researcher_reference/truth.npz` is the recovered 1.6 MB assessment bank,
kept separate from solver inputs. Its metadata records the initial field,
coefficient truth, times and original numerical qualification. The bank samples
a 256-square Fourier/DOP853 solve on a 64-square grid. Comparison with a
128-square solve gave a maximum framewise relative difference of `3.36e-9`,
below the declared `1e-7` threshold. This development reference was used in
previous assessments; it is not a new blind test.

To regenerate and qualify the reference in a fresh directory:

```bash
uv run python experiments/hj_inverse/reference.py --output /tmp/hj-reference
uv run python experiments/hj_inverse/evaluate.py \
  --reference-root /tmp/hj-reference --output /tmp/hj-regenerated-reference.json
```

The generator retains the original Fourier derivatives, DOP853 tolerances
(`rtol=1e-9`, `atol=1e-11`), maximum step `0.001`, and 128/256 grid comparison.
It performs two numerical PDE solves; ordinary checkpoint replay uses the
included reference and does not require regeneration.
