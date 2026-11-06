# Strategy - Allen-Cahn 2D Peanut Merger

A living account of how to attack this problem, maintained across generations of
workers. Read it before you start; revise it before you stop. Add what you
learned, prune what proved wrong, and sharpen what proved useful. This document
is how the search accumulates understanding: the directions below are informed
by earlier attempts, and what you record steers the workers who follow.

## What makes this problem hard

The initial condition is a connected, asymmetric peanut built from two
overlapping diffuse disks. The main difficulty is the neck: it is a genuinely 2D
interaction zone, not a pair of independent circular fronts. A model that simply
shrinks two disks can fit parts of the initial snapshot while missing the
curvature-driven relaxation into a smoother oval.

The evaluator also checks the initial time direction. Matching the value of the
peanut at `t = 0` is not enough; the one-sided tangent must be consistent with
`eps^2 Delta IC + IC - IC^3`. This makes purely geometric interpolation risky if
it has the right shapes but the wrong early velocity.

Finally, the domain is periodic. Opposite sides of the square must match in both
value and derivative, so a signed-distance ansatz should either be intrinsically
periodic or be corrected near the box edges without introducing a broad
black-box field.

Workflow constraint: do not use the fixed evaluator, Allen-Cahn reference bundle,
or `evaluate.py --json` as a local oracle while designing a candidate. Use
analytic reasoning, synthetic point checks, and the mandatory `check-runner`
instead. Also keep edits strictly inside the listed editable solver files and
the `guidance/` record; project metadata such as `pyproject.toml`, `uv.lock`, and
`*.egg-info` is not part of the solver structure.

## Directions

Starting points, not a closed list. If the running best has stalled, the search
has narrowed, not exhausted the space - open a new direction and record it here.

- A low-parameter signed-distance scaffold for the connected peanut, using the
  two diffuse disks as a smooth union at `t = 0`.
- A time-dependent blend from the peanut distance field to an oval or rounded
  connected interface, with separate controls for neck smoothing, axis lengths,
  and center drift.
- A curvature-aware early-time correction that targets the PDE tangent at
  `t = 0`, instead of only improving the initial value loss.
- Periodic coordinate features or boundary factors that make value and
  first-derivative matching natural on `[-1, 1)^2`.
- A compact secondary correction only after the geometry is explicit. Large
  Fourier patches should not dominate the submitted structure.

## What has been tried

Record here the structures that have been attempted, with the outcome: which
score dimension they satisfied or failed, and the mathematical reason. This
record is what prevents later workers from repeating dead ends, and it is the
evidence of how the strategy has sharpened over time. Append one entry per
attempt, newest last, in roughly this shape:

> *Submitted [the structure, with its key terms and trainable quantities].
> Intended to improve [which dimension]; a quick probe suggested [provisional
> result, if any]. The tradeoff was [what got worse, e.g. which dimension or
> region]. A later worker should try [the next concrete variant].*

*(No attempts recorded yet.)*
