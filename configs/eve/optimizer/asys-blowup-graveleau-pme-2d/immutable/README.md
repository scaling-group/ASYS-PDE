# Evolutionary Discovery Workspace

## EvE, ASYS and your role

ASYS uses [Evolving Ensemble of Agents (EvE)](https://arxiv.org/html/2605.09018v4)
to search over mathematical representations of PDE solutions. EvE maintains
populations of solver programs and research guidance; workers build on sampled
examples and evaluation feedback. ASYS combines revisions of the mathematical
form with numerical coefficient fitting. Where the task permits, workers may
also revise the training loss and fitting method.

You are an ASYS worker. Produce one scientifically defensible solver candidate
and improve the guidance in `guidance/` with reusable mathematical ideas, tests
and lessons. Read the task, inherited guidance and sampled examples before
choosing a change.

Work within your assigned workspace and read the supplied code to understand
its mathematical interfaces and evaluation contract. Use live web search for
public mathematical literature and cite any ideas you adopt. The sections below
define the scientific task, editable files, fitting interfaces, local checks
and official evaluation.

## The Problem

### Graveleau PME Focusing, m = 2, d = 2

Find a compact, trainable radial pressure field `v(r, t)` for the porous-medium
pressure equation

```text
v = 2 u
v_t = v (v_rr + (1 / r) v_r) + v_r^2
```

| | |
|---|---|
| Domain | `0.002 <= r <= 6` |
| Time horizon | `0 <= t <= 0.995 * 0.6200901156700502` |
| Initial data | centered hole: `v(r, 0) = 0` for `r < 1`, smooth positive shell outside |
| Geometry | radial 2D |
| Free boundary | pressure stays nearly zero inside a shrinking hole |
| Inputs | physical `(r, t)` |
| Output | one column, pressure `v` |

This is a focusing problem, not an amplitude blow-up problem. The hard part is
discovering a shrinking free-boundary law and a pressure profile near the moving
interface. The expected late regime is second-kind self-similar: the front
exponent is not determined by dimensional analysis alone. A naive first-kind
law such as `R(t) ~ sqrt(T - t)` can look plausible on early slices while
missing the near-focus residual.

The live evaluator is authoritative for the pressure convention, radial
autodiff convention, IC, outer Neumann boundary, training budget, helper API, and
reference-free score dimensions.

## Scoring

Two distinct optimizations are present and must not be conflated.

*Training*, inside your code, fits the trainable parameters of a fixed structure
by minimizing a scalar loss you define in `loss.py` / `config.yaml` / `optim.py`.
Its weights are yours to choose; it is a private means, not the objective being
judged.

*The evolutionary search* scores your structure on four independent dimensions,
computed on evaluator-owned held-out monitor points:

- `physics` - pressure-PME residual on uniform and front-biased monitor points;
- `ic` - value match to the hole-plus-shell initial pressure;
- `ic_tangent` - agreement of the initial-time secant over `h = 6.1699e-4` with
  `v (v_rr + v_r / r) + v_r^2` at the initial pressure, with the hole region
  remaining zero;
- `bc` - outer Neumann boundary loss at `r = 6`.

The evaluator reports the four signed components and their mean. Candidate
examples are sampled using their separate component ranks, allowing candidates
with different strengths to contribute. Training-loss weights do not change
these evaluation criteria. Diagnostics also report dimensionless residual,
interface behavior, hole pressure, and density mass; use them to spot
front-only fits.

## What Does Not Count As A Solution

Five classes of construction reproduce target values without representing the
mathematics and are rejected by the tangent dimension, static checks, or
semantic review.

A construction that **abandons the initial condition** - matching `v(r, 0)` and
then switching to an unrelated field for `t > 0`. The `ic_tangent` dimension
exists to reject this kind of time gate.

A **numerical solver in disguise** - integrating the pressure PME internally
with finite-difference, finite-volume, spectral, Runge-Kutta, ETD, split-step,
or other time-marching machinery and fitting or hard-coding the resulting
trajectory. Analytic self-similar forms, free-boundary ansatzes, asymptotic
profiles, and low-rank symbolic corrections remain admissible; marching the PDE
forward in time does not.

A **lookup table or reference leak** - a dense precomputed surface of values
keyed to grid points, private logs, or the hidden reference bundle. Do not read
hidden reference environment variables, local reference `.npz` files, generated
evaluator snapshots, or public-grid target arrays as a source of solution values.

A **model whose output is independent of its parameters** - at least one
trainable parameter is required, and the output must depend on it.

A **correction tuned to the evaluator's probe points** - a term whose location,
scale or cutoff is chosen to exploit evaluator sampling locations rather than
to represent the continuous PDE with its prescribed initial/boundary data.
For example, a cutoff chosen solely to retain an initial-tangent score while
hiding its transition from residual samples does not count. Judge the actual
field, including its transition regions, against the equation, initial data,
boundary conditions and domain. Localization or decay alone is not evidence
of gaming: genuine initial/boundary layers and other corrections justified by
the continuous problem are admissible. A localized correction need not improve
the residual everywhere or when extended throughout the domain.

## Information and representation rules

Solvers and checkers follow the same rule. Use the supplied task inputs,
constraints, approved observations, sampled guidance/candidates and public
mathematical literature. Construct explicit mathematical ansatze; do not use
neural solution surrogates, embedded numerical PDE solvers, trajectory tables,
or coefficients fitted to numerical solution samples.

Use public literature for mathematical ideas. Do not use withheld reference
solutions, private project artifacts or unsupplied fitted candidates as sources
of solution values.

## Editable Files

{editable_files_block}
{editable_folders_block}

The training loop and held-out scoring live in the fixed `evaluate.py`, which you
do not edit. The loss helpers include:

- `pde_residual(model, pts)` and `physics_loss(model, pts)`;
- `radial_derivative(model, pts)` and `radial_laplacian(model, pts)`;
- `outer_neumann_bc(model, pts)`;
- `initial_condition(r)`, `initial_condition_torch(r)`;
- `graveleau_rhs(v, v_r, lap)`;
- `as_scalar_field(out)` and `pressure_to_density(v, m=2)`;
- constants including `T_FOCUS_APPROX`, `T_MAX`, `R_MAX`, `R_EPS`,
  `HOLE_RADIUS`, `M_EXPONENT`, and `N_DIM`.

For this workspace, do not run the fixed evaluator or `evaluate.py --json` while
designing a candidate; the final runner owns official evaluation. Use code
inspection, analytic residual checks, synthetic point tests, and the mandatory
`check-runner` instead.

## Optimizer guidance

Optimizer guidance lives in the `guidance/` folder.

Guidance exposure:

- `guidance/skills/` is symlinked as `.agents/skills` and `.codex/skills`.
- `guidance/agents/codex/` is symlinked as `.codex/agents`.

The whole `guidance/` folder is editable, including ordinary guidance files,
`guidance/skills/`, and `guidance/agents/`, except the following read-only
immutable overlay files:

{immutable_overlay_block}

Use existing guidance skills and subagents when they help. When you learn
something reusable, update the most appropriate part of `guidance/`: ordinary
notes/docs for general strategy, `guidance/skills/` for reusable workflows or
skills, and `guidance/agents/` for specialist roles. These skills and agents
will be exposed through the provider symlinks.

Do not limit guidance updates to task facts. Some reusable guidance should be
self-prompts: any instructions that change how the agent works, including
working style, workflow, or guidance update principles themselves.

## Workspace

All paths below are relative to this workspace.

- `guidance/` - the research record (`strategy.md`, `references.md`). Read it before you begin. Revise it before you stop.
- `solver/` - the solver submission tree, prefilled from a reference solver. The
  editable files live here. Before changing them, read the fixed `evaluate.py`,
  the current `loss.py`, `optim.py`, and `config.yaml` to understand the
  scoring, the available helpers, and the current candidate. This is what is
  extracted as your solver candidate. Do not change project-root metadata such
  as `pyproject.toml`, `uv.lock`, or any `*.egg-info` files, and do not run
  setup commands whose only effect is to rewrite package metadata. Do not run
  `uv sync`, `pip install`, or ad hoc `uv run` commands from inside `solver/`;
  the predefined checker owns environment-backed validation.
- `logs/optimize/` - write working notes, intermediate artifacts, and debugging
  material here. It is preserved as run logs, separate from the submission.

If reference solvers are sampled for this run, read their public diagnostics
(`eval_report.txt`, `score.yaml`, prior `model.py` / `loss.py` / `optim.py`)
before deciding what to change. Do not inspect private reference data or
evaluator-owned target arrays.

## Reference Solver Examples

{solver_examples_block}

{optimizer_examples_block}

## How To Work

Aim for one structurally defensible candidate, not an open-ended search. Useful
starting points:

- Separate IC fidelity from front motion. Preserve the initial hole-plus-shell
  profile at `t = 0`, then add a smooth time factor whose derivative can match
  the pressure-PME initial tangent without leaking pressure into the hole.
- Parameterize the shrinking front explicitly. Use a bounded focus time,
  exponent, and interface coordinate such as `eta = r / R(t)`, but guard against
  collapse back to a first-kind law.
- Use second-kind structure: trainable exponent/profile terms, smooth
  interface-aligned bases, and a separate outer shell/plateau can be more
  meaningful than a single soft MLP over `(r, t)`.
- Keep the hole region nearly zero and nonnegative. Penalize hole leakage,
  negative pressure, or density-mass creation in the training loss if doing so
  helps, but do not use hidden reference values.
- Use your own synthetic points for algebraic and residual checks under
  `logs/optimize/`. Official evaluator diagnostics arrive after submission.

Then record what you learned. Advancing the research record is the second half of
this task. Before running the final checks, add a short entry under
"What has been tried" in `strategy.md` stating the structure you submitted, which
score dimension it was meant to improve, what any quick probe suggested, and
what a later worker should try next. If you mostly kept an inherited structure,
say which one and why. Leave `references.md` unchanged unless a live literature
search actually informed a structural choice, in which case record the source
and the specific lesson.

When both `solver/` and `strategy.md` are in order, invoke the predefined
`check-runner` sub-agent and have it run its configured checks from the workspace
root. Do not finish without it; if anything fails, repair and rerun until it
passes. Work autonomously - do not ask for clarification, approval, or feedback.
Write your summary and stop.
