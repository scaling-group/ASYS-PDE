# Allen-Cahn discovery workspace

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

Find a differentiable scalar field `u(x,y,t)` satisfying

```text
u_t = eps^2 (u_xx + u_yy) + u - u^3,    eps = 0.04,
(x,y) in [-1,1)^2,    0 <= t <= 8.
```

Positive-time dynamics are periodic in both spatial directions. The prescribed
initial field is the smooth union of two diffuse disks:

```text
u0 = (p1 + p2 + 1 - p1*p2)/2,
p_i = tanh((R_i - sqrt(|(x,y)-c_i|^2 + 1e-14))/(sqrt(2)*eps)),
c1 = (-0.15,0.02), R1 = 0.22;    c2 = (0.18,-0.03), R2 = 0.18.
```

The positive phase is already a connected, asymmetric peanut. Its overlap neck
is a two-dimensional interaction region. The tanh interface profile,
sharp-interface curvature motion, maximum principle and energy dissipation are
useful context, not a prescribed ansatz. The datum's off-center tails are not
smoothly periodic at the seam; preserve the stated datum and distinguish its
initial trace from positive-time boundary behavior.

The initial candidate is `u0 + (t/8)*a`, with `a=0` and no previously fitted
parameters.

Your candidate must be a **purely symbolic** mathematical expression.

**Allowed:**
- `nn.Parameter` quantities
- Explicit mathematical operations: `torch.exp`, `torch.tanh`, `torch.cos`,
  `torch.sqrt`, `torch.where`, `torch.clamp`, etc.
- Coordinate transformations, signed distance functions, level-set
  constructions
- Piecewise or conditional logic for regime transitions

**Forbidden:**
- `nn.Linear`, `nn.Conv1d`, `nn.Conv2d`, or any hidden-layer architecture
- MLP or fully connected networks of any depth or width
- Any module that learns a free-form mapping between activations

There is no parameter-count or parameter-naming restriction.

## Workspace

All paths are relative to this phase workspace, not the outer run directory.

- `solver/` contains your candidate, possibly prefilled from an earlier candidate
  in this run. Protected `problem.py` defines the equation, datum and helpers;
  protected `evaluate.py` performs official fitting and public scoring.
- `guidance/` carries inherited research guidance. Read it before you begin.
  Revise it before you stop. Immutable overlays remain read-only.
- `solver_examples/` and `guidance_examples/` contain sampled candidates and
  guidance. Read their source, evaluation logs and scores together to understand
  strengths, failures and reasons for earlier choices.
- `logs/optimize/` holds notes, trial logs and intermediate artifacts, preserved
  separately from the solver submission.

## Editable files

Within `solver/`, only the listed files and folders are editable:

{editable_files_block}
{editable_folders_block}

The list determines whether `loss.py`, `optim.py` and `config.yaml` may also change;
when only `model.py` is listed, fitting sources stay fixed. `solver/state/` holds
numerical checkpoint artifacts, not source code or fitting routines. Keep
protected equations, scoring and runtime unchanged. Do not import historical
fitted answers, external checkpoints or reference fields, embed trajectory
tables, march the PDE in the model, or fit inside its constructor, forward
method or helpers. Closed-form constructions and analytic coordinate maps
remain available.

## Fitting interface

`Candidate(config)` is a float64 `torch.nn.Module`: `forward` takes an `N x 3`
tensor of `(x,y,t)` and returns `N x 1` values supporting time derivatives and
second spatial derivatives. Fitted prediction quantities must be registered
parameters or buffers so `state_dict` reconstruction reproduces the field.

Where editable, `loss_fn(model, collocation, ic_points, bc_points, helpers)`
returns a scalar tensor and `build_optimizer(model, config)` returns a
`torch.optim.Optimizer`. The protected loop calls `optimizer.step(closure)`;
the closure evaluates the loss and backward pass. `helpers` is `problem.py`,
exposing `initial`, `initial_rhs`, `gradient`, `equation`, `terms` and `sample`.
Alternative loss terms, weights and reproducible sampling are allowed.
`config.yaml` controls the seed, batch counts and maximum steps. The initial loss
sums equation and constraint terms plus squared range excess; the initial numerical
optimizer is LBFGS.

## Scoring

Public scoring reconstructs the saved field in a fresh interpreter and reports
negative MSEs for four fixed checks:

- `physics`: strong PDE residual on 4096 uniform monitor points;
- `ic`: initial field on a 32 x 32 endpoint-free grid;
- `ic_tangent`: initial-time secant `(u(x,y,0.008)-u(x,y,0))/0.008`
  against `F[u0]` on a strictly interior 31 x 31 grid, excluding the
  incompatible periodic seam;
- `bc`: periodic values and common-coordinate normal derivatives on 128
  positive-time boundary pairs.

These fixed monitors provide search feedback, not unseen validation.
The scalar `score` is their equal mean; higher is better. The four components
also inform evolutionary selection. Training-loss weights
do not change these checks. The `terms` helper uses the exact AD initial
derivative for training; public scoring uses the finite-time secant above.
Range is a training term rather than a public admission gate. A better score
does not certify field or interface accuracy.

The official evaluator validates checkpoint source and config, then replays it
without fitting. Without a checkpoint it fits the constructor state.

## How to work

Read the current candidate, loss, optimizer and evaluator. Choose a mathematical
hypothesis worth testing, implement it through the editable interface, and use
the results to distinguish representation, fitting and coverage limitations.
Attribute constructions obtained from live literature. Finish with the structure
and term meanings, initialization, findings and remaining limitations. Use
GPT-6.1 SoL/xhigh/live web for any additional model-bearing role.

You may write and run your own fitting and diagnostic code under
`logs/optimize/`, using the editable candidate, loss and optimizer, and public
`problem.py` helpers on your own training and diagnostic points. Do not execute,
import, copy or rehost the fixed evaluator or its scoring routines, invoke them
through wrappers or other agents, or recreate its fixed public scoring as a
workaround. This applies even when inherited guidance or sampled logs show such
calls. Official evaluation runs after submission.

Use one CPU thread and a 300-second allowance for each fit. The official fallback
fit reserves 30 seconds for scoring, stops after `max(30 seconds, 30% of the
allowance)` without training-loss improvement, and retains the best evaluated
training-loss state. A single optimizer step or scoring can overrun the reserve.

To submit a fitted state, use `torch.save` to write `solver/state/checkpoint.pt`
as a dictionary with these fields:

- `state`: the model's `state_dict()`, including all prediction parameters and
  buffers;
- `effective_config`: the contents of `solver/config.yaml` parsed with
  `yaml.safe_load`;
- `source_sha256`: the mapping from `model.py`, `loss.py`, `optim.py`,
  `config.yaml`, `problem.py` and `evaluate.py` to each file's SHA-256 hex digest
  in `solver/`.

The state must match the submitted source and configuration. Hash protected
files without executing them. Keep each trial's editable source, config, state
and logs together. You may warm-start your own fitter from a prior local trial's
parameters and buffers with fresh optimizer history. To restore an earlier
trial, restore its matching editable source, config and state. After changing
source or config, refit and export a matching
checkpoint. Do not edit source during a fit. Use `-B` for Python checks to avoid
protected bytecode changes.

Before running the final checks, update
`guidance/` with the structure you submitted, which score dimension it was meant
to improve, what any quick probe suggested, and what a later worker should try
next. If you mostly kept an inherited structure, say which one and why.

Sampled solver and guidance examples:

{solver_examples_block}
{optimizer_examples_block}

Immutable guidance overlays:

{immutable_overlay_block}

Paper pointer: https://arxiv.org/html/2605.09018v4 (EvE).
