# References - Graveleau PME Focusing

Persistent references and known facts for this workspace. Keep this short and
specific: add sources only when they directly inform a structural choice.

## Known Facts

- The live evaluator is authoritative for the pressure convention, radial
  derivatives, helper API, hidden monitor points, training budget, and score
  dimensions.
- The benchmark uses pressure `v = 2u` and solves
  `v_t = v (v_rr + v_r / r) + v_r^2` on radial 2D geometry.
- The domain is `0.002 <= r <= 6` and `0 <= t <= 0.995 * 0.6200901156700502`.
  The initial pressure is zero for `r < 1` and a smooth positive shell outside.
- The evaluator scores `physics`, `ic`, `ic_tangent`, and `bc` independently.
  The `ic_tangent` target is computed from the pressure-PME RHS at the initial
  pressure, with the hole region clamped to zero.
- Do not read hidden reference data or use it as design data.

## Blow-up PDE Challenges and Techniques

- Pure residual fitting can prefer trivial, flat, or badly scaled branches on
  focusing and blow-up problems. The pressure variable has a free-boundary
  structure, so a small raw residual can still miss the focusing profile if the
  amplitude or interface scale is unconstrained.
- A normalization condition is the standard scale anchor in recent self-similar
  blow-up searches. Fix one nonzero pressure/profile value, derivative value, or
  interface-local normalization. This is not hidden-reference supervision; it is
  a gauge choice that keeps the residual problem from collapsing to the zero or
  rescaled branch.
- For Graveleau-type PME focusing, useful structural guesses should work in the
  evaluator's pressure convention `v = 2u` and similarity coordinates near the
  moving interface. A typical template is a profile
  `v(r,t) = scale(t) * Phi(eta)` with `eta` tied to the focusing length scale;
  the exact exponents are part of the structure to infer or parameterize.
- The initial shell, zero hole region, and outer Neumann boundary are all part of
  the evaluator contract. Do not read the hidden reference bundle, and do not use
  time marching; use analytic/profile structure plus evaluator-owned residuals.

## Sources

- Yongji Wang et al., "Discovery of Unstable Singularities", arXiv:2509.14185,
  https://arxiv.org/abs/2509.14185.
- Yongji Wang, Ching-Yao Lai, Javier Gómez-Serrano, and Tristan Buckmaster,
  "Asymptotic self-similar blow-up profile for three-dimensional axisymmetric
  Euler equations using neural networks", arXiv:2201.06780,
  https://arxiv.org/abs/2201.06780.
- "When PINNs Go Wrong: Pseudo-Time Stepping Against Spurious Solutions",
  arXiv:2604.23528, https://arxiv.org/abs/2604.23528.
- Rafael Leiteritz and Dirk Pflüger, "How to Avoid Trivial Solutions in
  Physics-Informed Neural Networks", arXiv:2112.05620,
  https://arxiv.org/abs/2112.05620.
- Aditi S. Krishnapriyan et al., "Characterizing possible failure modes in
  physics-informed neural networks", arXiv:2109.01050,
  https://arxiv.org/abs/2109.01050.
- Local evaluator authority:
  `solver/problems/blowup/graveleau_pme_2d/evaluate.py`
  referenced by
  `configs/eve/application/asys-blowup-graveleau-pme-2d.yaml`.
