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

## The problem

### Hamilton–Jacobi inverse problem

Construct an explicit differentiable field and infer four independent physical
coefficients from sparse noisy observations on the periodic square `[0,2*pi)^2`:

```
u_t = d_x*u_xx + d_y*u_yy + b_x*u_x^2 + b_y*u_y^2.
```

`solver/problems/inverse/hj_coefficients/problem.json` supplies the fixed Fourier
initial field, normalization, horizon `H=34/98`, coefficient bounds and starting
guesses. The prescribed initial field is:

```
u0(x,y) = sum(amplitude*sin(kx*x + ky*y + phase)
              / ((abs(kx) + abs(ky) + 1)*initial_scale)).
```

The four coefficients are independent. A common nonlinear/diffusive ratio is
not supplied and must not be assumed. Define the field through `t=0.5`, beyond
fitting and validation. No accuracy beyond `H` is assessed by the public score.

The approved `observations.npz` beside `problem.json` contains only
`train_points`, `train_values`, `validation_points`, `validation_values`.
Points are float64 `[N,3]` physical `(x,y,t)` and values `[N,1]`.
Training has 5,945 observations at frames 1–29; validation has 1,025 at frames
30–34. Fit on training; use validation for public checkpoint selection and
validation. `OBSERVATIONS.md` records the observation packet.
These supplied observations are permitted training inputs. Clean fields,
coefficient truth, future reference fields and other-run artifacts are withheld.
Do not import fitted states or old populations.

## Scoring

The public score is minus the sum of normalized equation, initial, periodic and
observation-validation losses. Physics is scaled by `H/field_scale`, values by
`1/field_scale`, periodic derivatives additionally by `2*pi`. Four fixed
validation batches and all validation observations select the earliest minimum
at step zero, every 500 updates and the final step. The selected full state,
including physical coefficients, is saved and replayed. Public loss does not
measure coefficient recovery or prove identifiability.

## Information and representation rules

Solvers and checkers follow the same rule. Use the supplied task inputs,
constraints, approved observations, sampled guidance/candidates and public
mathematical literature. Construct an explicit field and fit the four bounded
physical coefficients using the supplied observations and equation constraints.
Do not use neural surrogates, candidate-embedded numerical PDE solvers,
trajectory tables or withheld numerical reference solutions.

Use public literature for mathematical ideas. Do not use withheld reference
solutions, private project artifacts or unsupplied fitted candidates as sources
of solution values.

## Editable files

{editable_files_block}
{editable_folders_block}

Under `solver/`, edit only the four declared `model.py`, `loss.py`,
`optim.py`, `config.yaml` files.
`Candidate(config): torch.nn.Module` receives the public problem fields.
`forward(points)` returns float64 `[N,1]`, is pointwise, has first x/y/t and
second spatial derivatives. Initial and periodic value/first-normal-derivative
compatibility enter the fixed public loss.
The starting field is the unfitted Fourier initial field plus a damped
first-order correction, with four zero amplitudes and an untrained decay scalar.

The evaluator owns four bounded physical parameters. Optional
`forward_with_coefficients(points, coefficients)` receives their differentiable
`(d_x,d_y,b_x,b_y)` vector; otherwise field and physical parameters are distinct.
The diffusion bounds are `(0.01,0.3)` and nonlinear bounds `(-2,1)`, with start
`(0.06,0.20,-0.3,-0.7)`.

Construct an explicit mathematical expression, analytic expansion or transform.
Do not use neural fields, numerical PDE time marching, interpolated trajectory
tables, hidden reference data or a fitting procedure inside `Candidate`.
Spatial projection/quadrature of an analytic expression is permitted. Explain
parameter roles, approximations and validity in source. The evaluator supplies
approved observations to fitting; candidate files do not load external data.

`loss.py` defines `training_loss(components, config)` and `optim.py` defines
`make_optimizer(parameters, config)`. Components are `physics`, `ic`, `bc`, and
`observation`. `config.yaml` selects steps, learning rate and batch counts.
The maxima are 20,000 updates, 256 physics, 128 initial, 64 periodic
pairs and 256 training observations per update. The initial recipe is Adam at
0.001. `step(closure)` uses one fixed minibatch per update, allowing Adam, SGD
or L-BFGS. The production guard is 6,600 seconds, checked at update
and closure entry; an exhausted official fit fails. Float64, seed 1101 and one
CPU thread are fixed. Setup and terminal checks may add elapsed time.

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

- `guidance/` — optimizer guidance files.
- `solver/` — the solver submission tree, prefilled from a reference solver. The
  editable files live here. Before changing them, read the fixed `evaluate.py`,
  the current `loss.py`, and `config.yaml` to understand the scoring, the
  available helpers, and the current candidate. This is what is extracted as your
  solver candidate.
- `logs/optimize/` — write working notes, intermediate artifacts, and debugging
  material here. It is preserved as run logs, separate from the submission.

If reference solvers are sampled for this run, read their diagnostics
(`eval_report.txt`, `score.yaml`, prior `model.py` / `loss.py`) before deciding
what to change.

Reference solvers sampled for this run, with their scores:

{solver_examples_block}

{optimizer_examples_block}

## How to work

Aim for one structurally defensible candidate, not an open-ended search. Use
code inspection and analytic or synthetic checks under `logs/optimize/` while
designing the field. The predefined `check-runner` performs scoped interface,
admissibility and boundary checks using the prepared interpreter; its quick
qualification does not replace the official fit. Once you have a candidate
worth submitting, write it into the editable files under `solver/`.

Then record what you learned. Advancing the research record is the second half of
this task, not optional cleanup. Before running the final checks, update
`guidance/` with the structure you submitted, which score dimension it was meant
to improve, what any quick probe suggested, and what a later worker should try
next. If you mostly kept an inherited structure, say which one and why.

When both `solver/` and `guidance/` are in order, invoke the predefined
`check-runner` sub-agent and have it run its configured checks from the workspace
root. Do not finish without it; if anything fails, repair and rerun until it
passes. Work autonomously — do not ask for clarification, approval, or feedback.
Write your summary and stop.
