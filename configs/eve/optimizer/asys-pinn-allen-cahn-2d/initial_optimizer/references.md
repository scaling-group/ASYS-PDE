# References - Allen-Cahn 2D Peanut Merger

Accumulated literature and known results that inform a structural choice for this
problem, maintained across generations of workers. This document starts sparse;
append entries as external results become relevant, each with a one-line note on
how it applies to the structure being built.

## Known results

- Allen-Cahn diffuse interfaces are naturally represented by tanh profiles across
  an `O(eps)` transition layer. This supports signed-distance coordinates rather
  than dense surface fits.
- In the sharp-interface limit, Allen-Cahn interfaces move approximately by mean
  curvature. For this case, the overlapping peanut neck is therefore part of the
  dynamics, not just an artifact of the initial condition.

*(Append further entries here: interface motion results, compact peanut-to-oval
parameterizations, or periodic phase-field representations, each with a note on
how it informs the model structure.)*
