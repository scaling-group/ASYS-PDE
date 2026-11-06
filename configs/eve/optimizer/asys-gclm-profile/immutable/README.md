# Real-line gCLM similarity profile

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

Construct an explicit, differentiable odd profile, positive for y>0, for the real-line
gCLM similarity equation at a=.25:

    f + beta*y*f_y + a*lambda*V*f_y - lambda*g*f = 0
    g = -H_R[f], V'=g, V(0)=0, lambda=(1+beta)/(1-a).

The canonical gauges are f(0)=0, f'(0)=g(0)=1, beta>0. Seek a decaying,
positive-half-line profile with finite weighted normalization moment. This is one parameter value, not a
periodic trajectory or across-a family. Known mathematical symmetries,
Hilbert identities, compactification and the leading power envelope with an
adjustable beta are supplied mathematical starting points.

Only `model.py` is editable. It must expose `build_model(config)` returning a
CPU float64 torch.nn.Module, `forward(x)` mapping a 1-D real tensor to the same
shape, and `beta()` returning a positive finite scalar tensor. `forward(x)`
returns the raw profile h(x), from which the evaluator constructs the normalized
profile f below. All numerical
parameters/buffers required for prediction must be registered and replayable.
Use explicit analytic expressions with named mathematical parameters. Neural
networks, trajectory tables and numerical equation solvers as candidate
representations are forbidden. No expression-slot or parameter-count cap is imposed.
Report complexity rather than inferring efficiency from parameter count.

The raw h must be odd, zero at the origin with finite positive derivative
k=h'(0), positive for x>0, and have finite positive weighted moment
A=(2/pi)*integral_0^infinity h(x)/x dx. Protected
evaluation measures A and k once after fitting and freezes f(y)=h((A/k)*y)/A.
It does not repair a nonzero origin, negative profile, bad slope or unresolved
mass. Inference may evaluate this saved explicit representation; it may not fit,
solve the defining PDE/integral equation, normalize afresh, load reference tables
or use probe-dependent corrections. Parameterless explicit constructions have
a declared no-fit validation/export pathway and remain eligible if qualified.

Admissibility uses this finite weighted moment,
not a new ordinary half-line L1 requirement. The beta=1 starter has finite A
and divergent ordinary L1 mass. Finite mesh/operator checks certify their
instrument resolution only, not global decay or full-line integrability.

Protected fitting uses 20,000 Adam updates, initial lr .01 with cosine schedule;
the last complete measured update is retained. It has a 1200-second allowance
including validation reserve 30s; the whole evaluation is capped at 1500s.
After numerical resolution checks pass, evaluation computes the logarithmic-radius
mean of (sum T_i)^2 / sum(T_i^2), where T_i are the four terms of the similarity
equation. The public score is
-(residual_mean + f(0)^2 + (f'(0)-1)^2 + (g(0)-1)^2); higher is better.

Use the supplied candidate source, guidance and evaluation logs to inform revisions.
Keep any initialization assumptions in model source explicit. The starter
uses the supplied unfitted envelope, beta initialized at 1, and zero correction.
No reference beta or reference fields may initialize candidates or enter fitting.

## How to work

Do not edit or execute protected evaluation while acting as a source proposer.
Small local algebra/interface checks within the session allowance are permitted;
formal fitting and judging belong to native evaluation. Do not inspect other
runs or private reference data. Live primary
literature research is available; attribute mathematical priors and reused ideas.

Finish one coherent candidate, explain its structural hypothesis, and preserve
uncertainty and adverse public diagnostics.

Before running the final checks, update
`guidance/` with the structure you submitted, which score dimension it was meant
to improve, what any quick probe suggested, and what a later worker should try
next. If you mostly kept an inherited structure, say which one and why.

## Workspace

All paths are relative to this workspace. `solver/` contains the candidate and
protected mathematical helpers; `logs/optimize/` is for local notes and checks.

Optimizer guidance lives in the `guidance/` folder.
Read it before you begin. Revise it before you stop.

{editable_files_block}
{editable_folders_block}

Public solver examples:
{solver_examples_block}

Guidance examples:
{optimizer_examples_block}

{immutable_overlay_block}

Read the actual source under solver/ and public feedback under solver_examples/.

Numerical coefficient fitting, Hilbert-transform quadrature and normalization
are owned by the protected evaluator. Candidate construction initializes the
explicit formula; it must not solve the profile equation or fit coefficients.
