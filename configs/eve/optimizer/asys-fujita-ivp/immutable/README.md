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

### Radial Fujita equation (n = 5, p = 3)

$$u_t = u_{rr} + \frac{4}{r}u_r + u^3$$

| | |
|---|---|
| Domain | r ∈ [0, 5] |
| Time horizon | 0 ≤ t ≤ 0.020385594 |
| Initial condition | u(r, 0) = 5 cos(πr / 10) |
| Boundary | u_r(0, t) = 0, u(5, t) = 0 |
| Inputs | physical (r, t) |
| Output | one column, u |

Construct and explain the time evolution, preserving the initial and boundary
constraints while resolving the late concentrating core and the outer field.
The regular origin diffusion is 5*u_rr. The endpoint is .999*.020406;
.020406 defines the evaluation horizon and is not a certified blow-up time.
A candidate clock may be fitted from the public constraints while keeping the
field finite throughout the interval. Do not assume a nonconstant stationary
similarity profile is this initial condition's concentrating core.

The objective is the mathematical structure. The training loop fits the numbers.
Write a model whose parameters are meaningful quantities and whose code reads as
a statement about the singularity, not a black-box approximation.

## Scoring

Two distinct optimizations are present and must not be conflated.

*Training*, inside your code, fits the trainable parameters of a fixed structure
by minimizing a scalar loss you define in `loss.py` / `config.yaml`. Its weights
are yours to choose; it is a private means, not the objective being judged.

*The evolutionary search* scores your *structure* on four independent dimensions,
computed on evaluator-owned held-out monitor points:

- `physics` — negative full-domain geometric mean of max(abs(R)/D,1e-12);
- `ic` — negative prescribed initial-value MSE;
- `ic_tangent` — negative MSE against the exact initial PDE time derivative;
- `bc` — negative mean of origin Neumann and outer Dirichlet MSE.

Here R=u_t-(u_rr+4*u_r/r)-u^3 and
D=max(abs(u_t)+abs(u_rr+4*u_r/r)+abs(u^3),1e-12).
The evaluator reports the four signed components and their mean. Candidate
examples are sampled using their separate component ranks, allowing candidates
with different strengths to contribute. Training-loss weights do not change
these evaluation criteria.

The global monitor uses 2048 LHS physics points, 512 initial radii including both
endpoints, 256 times per boundary, seed 920614. A separate core diagnostic is
mean((R/D)^2), with 2048 LHS points: eta uniform(0,10), tau=.020406-t log-uniform
between .001*.020406 and .1*.020406, r=eta*sqrt(tau), plus 128 origin points,
seed 920811. It is not an additional ranking dimension. The clock defines
coverage; it is not a physical law imposed on the field.

The initial tangent includes r=0, where F[u0]=125-pi^2/4. It excludes r=5, where
F[u0]=-2*pi/5 conflicts with the fixed boundary's time derivative. An initial
corner layer is legitimate; joint corner smoothness is not required. The same
continuous field must satisfy the equation and constraints independently of
scoring locations. Public equation residuals do not certify field or
radial-gradient accuracy.

## Information and representation rules

Solvers and checkers follow the same rule. Use the supplied task inputs,
constraints, sampled guidance/candidates and public mathematical literature.
Construct an explicit mathematical field for this initial-value problem.
Do not use neural surrogates, candidate-embedded numerical PDE solvers,
trajectory tables or coefficients fitted to withheld numerical solutions.

Use public literature for mathematical ideas. Do not use withheld reference
solutions, private project artifacts or unsupplied fitted candidates as sources
of solution values.

## Editable files

{editable_files_block}
{editable_folders_block}

The training loop and held-out scoring live in the fixed `evaluate.py`, which you
do not edit. Read `solver/problems/blowup/fujita_radial_n5_p3/problem.py` and
`evaluate.py` before changing the candidate.

`Candidate(config=None)` is a CPU float64 torch module. `forward(points)` maps
physical `(r,t)` points of shape `[N,2]` to pointwise `[N,1]` values, with first
time/radial and second radial derivatives. At least one trainable scalar must
affect the field. There is no parameter cap. The unfitted two-scalar
concentrating-cosine seed enforces the initial field and boundaries, with
trainable clock margin and spatial width. Its initial clock is .020406. It
cannot express the negative initial evolution near the outer wall.

Use explicit differentiable expressions, analytic transforms, series, physically
derived coordinates or declared mathematical quadrature. The same field must
work at arbitrary points independently of batch composition. Do not hide fitting
inside the representation or use numerical PDE marching/shooting.

`loss_fn(model,collocation,ic_points,bc_points,helpers)` supplies a finite
differentiable scalar. `build_stepper(params,helpers)` supplies `step(closure)`
and `zero_grad()`. Helpers expose `pde_terms`, `physics_loss`, `initial`,
`initial_rhs`, `core_points`, `loss_terms` and `boundary_loss`; inspect their
protected definitions. The config sets Nf/NIC/NBC and max_steps. The training
objective uses normalized squared residuals and is editable; it cannot change
the protected score. The official fit allows 300 elapsed seconds on one CPU
thread, retains the lowest finite training-loss state and uses the shared
stagnation stop.

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

Aim for one structurally defensible candidate, not an open-ended search. You may
run the fixed `evaluate.py` locally to gauge a direction, but keep such probes
short and few — two at most, each with a wall-time or sample cap — and treat
their scores as provisional, since the official score is computed only after you
stop. Once you have a candidate worth submitting, write it into the editable
files under `solver/`.

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
