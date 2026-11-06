# Generated fourth harmonic

This numerical experiment tests the manuscript's predicted coupling from a
second-harmonic interface perturbation to a fourth harmonic:
`e4=(11/20) a² (s²-s⁷)`, with `s=1-2 epsilon² t/R0²` and `R0=0.28`.
The prediction has no fitted coefficient.

`measurements/` contains all six primary PDE cases and both resolution checks.
Each has settings, 61-time Fourier/area measurements and the complete sampled
zero contours. `curvature-bridge/` contains the independent polar mean-curvature
evolution and its refinement. The source supplies signed Euclidean initial
distance, Fourier ETDRK4 evolution and quintic-interpolated contour extraction.

Read the saved outcomes without solving the PDE:

```bash
uv run python experiments/allen_cahn/fourth_harmonic/evaluate.py --output /tmp/ac-fourth-harmonic
```

At `s=0.7`, the fourth-harmonic discrepancy is about `1.82–1.96%` for
`epsilon=0.02` and `6.08–6.18%` for `epsilon=0.04`, across amplitudes
`a=0.04,0.08`. Circular controls measure spurious fourth harmonics. Prescribed
`s` is the primary comparison; substitution of measured area remains a separate
diagnostic. The finite-epsilon discrepancies do not establish an exact PDE law.

To regenerate one case into a fresh directory:

```bash
uv run python experiments/allen_cahn/fourth_harmonic/source/run_pde.py \
  --case eps02-a08 --a 0.08 --eps 0.02 --n 512 --dt 0.00390625 \
  --output /tmp/ac-eps02-a08
uv run python experiments/allen_cahn/fourth_harmonic/source/curvature_bridge.py \
  --output /tmp/ac-curvature-bridge
```

The other primary cases use `a=0,0.04,0.08` and `epsilon=0.02,0.04`, with the
same 512-square grid and maximum time step `1/256`. The `epsilon=0.02,a=0.08`
refinements change only the grid to 768-square or the maximum step to `1/512`.
Full diffuse-field snapshots can be regenerated; the release includes the much
smaller complete contour records used for the mode assessment.
