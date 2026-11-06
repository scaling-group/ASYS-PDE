# Transient radial Fujita field

The selected formulas approximate the five-dimensional radial equation
`u_t = u_rr + 4 u_r/r + u^3` with initial data `5 cos(pi r/10)` and a zero
boundary value at `r=5`. Their initial time compatibility and boundary response
are encoded in the representation. `selected/L/` uses the spatially local
reaction-clock response; `selected/U/` uses the uniform response. Both include
the original model, configuration, 14 fitted coefficients and fitting definitions.

`response_swap/` supplies the shared initial state and the two fitted states.
The readout applies each raw state to both response definitions without refitting.
This isolates the response formula's effect from coefficient adjustment:

| Coefficients | Local response | Uniform response |
| --- | ---: | ---: |
| Initial | 0.07975% | 0.29691% |
| Local fit | 0.08793% | 0.29170% |
| Uniform fit | 3.99072% | 3.99071% |

These are relative full-trajectory field errors. The selected fits were chosen
by their training objectives; the reference error is an assessment of those
states and does not select a replacement.

Each response-swap row also reports the normalized equation MSEs on the
original 512 uniform and 208 core fitting
points (seeds 107014 and 107015), and on the independent neural-control sample:
8,192 uniform and 8,704 core points, including 512 at the origin (seeds
2026110596 and 2026110597). The residual is
`rho = (u_t - d - u^3)/max(|u_t| + |d| + |u^3|, 1e-12)`, with
`d = u_rr + 4 u_r/r` and its regular origin limit. Each objective is the uniform
mean of `rho^2` plus `0.2` times its core mean. Sampling seeds, diagnostic
values and source provenance are recorded in `response_swap/expected.json`.

`neural/` contains the current 12,739-parameter neural control and paired
initializations for three seeds. It multiplies the supplied concentrating seed
by a neural exponential correction. The main dense setting uses 5,760 points:
4,096 uniform and 1,664 core points. The supplementary sparse setting uses the
same first block of 512 uniform and 208 core points. The core set includes the
origin. All three 600-second and final selected checkpoints are included for
each sampling setting. Final errors are `6.01%, 5.30%, 5.01%` for dense sampling
and `7.90%, 6.09%, 6.33%` for sparse sampling.

From the repository root:

```bash
scripts/prepare_references.sh fujita
uv run python experiments/fujita/evaluate.py \
  --reference references/fujita/fujita-forward.npz --output /tmp/fujita.json
```

Use `--cases selected`, `--cases response-swap`, or `--cases dense sparse` for a
subset. Assessment integrates with `dr dt` on the original 3,549-radius,
289-time grid. The secondary core metric integrates the same energy densities
over `0 <= r <= 0.5`, including the last partial cell. `expected.json` and
`response_swap/expected.json` retain the saved readouts.

Optional refitting uses a fresh output directory:

```bash
uv run python experiments/fujita/neural/production_runner.py \
  --seed 2026110501 --sampling dense --out /tmp/fujita-neural-refit
```

Use `--sampling sparse` for the paired low-density setting. The 3,600-second
process allowance includes initialization and finalization. Adam runs until
2,400 seconds with a cosine learning rate from `0.001` to `0.0001`, followed by
L-BFGS. The objective is uniform residual loss plus `0.2` times core residual
loss. Selection retains the best completed step; a 600-second checkpoint must
have completed by that deadline. Saved training summaries identify the selected
steps. The reference field is never used by this fitter.
