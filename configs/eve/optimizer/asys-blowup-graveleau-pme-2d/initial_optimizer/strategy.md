# Strategy - Graveleau PME Focusing

A living account of how to attack this problem, maintained across generations of
workers. Read it before you start; revise it before you stop. Add what you
learned, prune what proved wrong, and sharpen what proved useful.

## What Makes This Problem Hard

This is a focusing problem in pressure variables, not an amplitude blow-up
problem. The pressure is zero inside a centered hole, and the moving interface
shrinks toward the origin as focus approaches. A model that only grows amplitude
or fits the outer shell misses the central free-boundary mechanism.

The late regime is second-kind self-similar. A first-kind front law such as
`R(t) ~ sqrt(T - t)` is a tempting starting point because it aligns with the
evaluator's front-biased sampling, but the actual exponent/profile are not fixed
by dimensional analysis. Letting focus time, exponent, and shape all float
without constraints can also collapse into a barely moving or first-kind-like
fit.

The evaluator scores `physics`, `ic`, `ic_tangent`, and `bc` independently.
Training weights are only a fitting device; a good structure should be
defensible on all four dimensions separately. The `ic_tangent` score compares
against the pressure-PME RHS at the initial hole-plus-shell pressure, with the
hole region remaining zero.

The interface is the hard region. Hole leakage, negative pressure, mass creation,
or a front-only profile with dimensionless residual near order one should be
treated as a failed mathematical structure even if one scalar loss improves.

Workflow constraint: do not run the fixed evaluator, `evaluate.py --json`, read
hidden reference bundles, or run ad hoc environment commands while designing a
candidate. Use analytic reasoning, synthetic point checks, direct source
inspection, and the mandatory `check-runner` instead. Keep edits strictly inside
the listed editable solver files and the `guidance/` record; project metadata is
not part of the solver structure.

## Directions

- Start from the exact initial hole-plus-shell pressure. Preserve zero pressure
  in the hole at `t = 0`, and add a tangent term that respects the pressure-PME
  RHS instead of an arbitrary time gate.
- Build the candidate around an explicit front `R(t)`, an interface coordinate
  like `eta = r / R(t)`, and a compact profile in `eta`.
- Keep focus-time and exponent parameters bounded. If the fit pushes focus time
  far away or the exponent back toward a naive first-kind law, record that as a
  failed direction rather than chasing scalar loss.
- Separate inner-hole, interface, and outer-shell behavior. Smooth barriers,
  softplus pressure, profile bases, and outer Neumann tapers can help maintain
  structure without lookup tables.
- In `loss.py`, consider private penalties for hole pressure, negative pressure,
  density-mass drift, and IC tangent only if they improve training stability.
  They are diagnostics and regularizers, not hidden target data.
- Keep probes synthetic and local to `logs/optimize/`: sample the candidate,
  inspect front location, hole leakage, pressure sign, and simple residual terms
  without invoking the fixed evaluator.

## What Has Been Tried

Record here the structures that have been attempted, with the outcome: which
score dimension they satisfied or failed, and the mathematical reason. Append
one entry per attempt, newest last, in roughly this shape:

> *Submitted [the structure, with its key terms and trainable quantities].
> Intended to improve [which dimension]; a quick probe suggested [provisional
> result, if any]. The tradeoff was [what got worse, e.g. which dimension or
> region]. A later worker should try [the next concrete variant].*

*(No attempts recorded yet.)*
