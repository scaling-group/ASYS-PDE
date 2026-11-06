# Reshaping a concentrating Keller–Segel density

The five saved fields share the Gaussian initial density
`u(r,0)=9.5 exp(-r²)` and radial mass `4.75`. The Gaussian mixture supplies the
fitted starting point for two separate revisions: reweighting its fitting
residual, or adding squared fourth-degree Laguerre factors to reshape its
components. The shape revision preserves nonnegativity, analytic enclosed mass
and the prescribed second-moment law.

`selected/gaussian/`, `selected/reweighted/` and `selected/laguerre4/` contain
their original model, fitting definitions, configuration and frozen state.
Configurations retain inherited coefficient values where the continuation used
them. The reported relative density errors are approximately `0.041%`,
`0.0063%` and `0.0058%`, respectively, in the norm weighted by `r dr dt`.
The illustrated fourth-degree field was selected by its density error in the
subsequent reference assessment. The equation-based search score ranked the
higher-degree `L4/L8` variant in `selected/laguerre48/` first; its subsequent
density error was `0.00913%`.

`selected/gaussian_continued/` contains the unchanged-family control: it starts
from the fitted Gaussian coefficients and gives the same objective another
300 seconds. The completed fit took 296 steps and 1,761 closures, moved no
coefficient by more than `7.90e-16`, and retained the `0.04095%` density error.
Its constructor differs from the Gaussian source only in the active coefficient
initializer. This control separates additional fitting from the changes in
residual weighting and radial shape. `expected.json` records all five outcomes;
the two added controls include the original field-and-derivative probes.

From the repository root, assess a cumulative-mass reference:

```bash
uv run python experiments/keller_segel/reference.py --output references/ks-192k.npz
uv run python experiments/keller_segel/evaluate.py --reference references/ks-192k.npz --output /tmp/ks.json
```

Reference generation is a large BDF solve: 192,000 radial cells and 77 times on
`[0, 0.86652]`. Use `--cells 96000` and a different output file for the recorded
spatial sensitivity comparison. Density is reconstructed from the native mass
derivative before selecting the common 48,001-node readout grid. This finite
reference comparison does not certify blow-up or a continuum error bound.

`--physics` evaluates the continuous residual on the original 36-time,
1,280-cell monitor, independently of reference data. It can be used alone.
`diagnostics.py` also provides the original independent mass, moment and flux
quadrature for inspecting the saved fields.

The search config `keller-segel-radial-2d` remains a fresh starting point. Its
zero coefficients describe an evolving core–halo field with distinct component
rates, Gaussian initial data and compatible time jets. The fitted continuation
states above belong to the manuscript example, not to every new search.
