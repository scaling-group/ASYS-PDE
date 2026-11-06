# Conditional Graveleau waiting-profile construction

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

This task studies a conditional similarity limit of porous-medium pressure
near a stationary inner boundary at r=1. Write p(r,t) for the pressure, governed
by p_t=p*(p_rr+p_r/r)+(p_r)^2, and tau for a hypothesized end of waiting.

Construct a useful pointwise profile F(xi) for
F*F''+(F')^2-(7/3)*xi*F'+(11/3)*F=0 on 0<=xi<=.01,
with F(0)=F'(0)=0, F''(0)=1/3, F'''(0)=6 and F,F'>=0.
In the local expansion F(xi)=xi^2/6+C*xi^3+..., the cubic normalization chooses
C=1. This is a conditional local component,
not a full radial IVP solution, a physical release time or a focusing exponent.

The profile equation follows formally from p=s^(2*gamma-1)F(q/s^gamma),
s=tau-t, gamma=7/3, q=r-1, under stationary regular waiting and a single-scale
limit with compatible differentiated remainders. The radial correction is
subleading under those assumptions. None establishes that actual waiting
persists to tau or that this profile describes the original IVP.

The supplied model.py contains an analytic Taylor starter through order seven.
The supplied guidance/theory.md contains a provisional higher-order resonance
calculation.

Challenge the calculation and its assumptions.
Investigate a materially different useful representation (for example weaker
regularity or more than one scale), rather than refitting the polynomial only.
You may research other approaches, reject priors with evidence, and revise the
representation, loss and fitting method. Do not force a favorable conclusion.
Document your own mathematical construction and its limitations in model.py's
docstring and retained logs.

## Workspace

All paths are relative to this workspace. `solver/` contains the candidate and
protected task helpers; `logs/optimize/` is for local notes, construction trials
and synthetic checks.

Optimizer guidance lives in the `guidance/` folder.
Read it before you begin. Revise it before you stop.

Within `solver/`, only model.py, loss.py, optim.py and config.yaml are editable.
Candidate() returns a torch.nn.Module whose forward method takes an N-by-1
tensor and returns N-by-1 values. Registered state
must restore all persistent learned/fixed values. Use explicit analytic factors, named mathematical parameters and
differentiable quadrature. Neural-network layers are forbidden. No PDE/ODE marching,
shooting, coefficient fitting, external references or scorer access in forward.
Do not change task.py, check.py or evaluation. Avoid generated application cache
files; the baseline .gitignore is fixed. Local construction/self-fit work belongs
in `logs/optimize/` and consumes this session's 2700-second allowance.

## Numerical fitting and scoring

The protected fitter has a cooperative 300-second cap and selects eligible
minimum-training-loss checkpoints in a fresh interpreter. Public scoring uses
fixed independent Sobol grids, normalized ODE residual, zero trace and supplied
quadratic/cubic normalization. Scores guide this profile construction;
they do not certify asymptotics. There is no numerical-reference fitting target.
A pure quadratic solves the ODE but is excluded by the nonzero cubic constraint.
Admit the distinction between approximating finite windows and realizing a
consistent limit at xi=0; finite eighth derivatives from a branch in AD do not
prove an analytic profile.

## How to work

Before running the final checks, update
`guidance/` with the structure you submitted, which score dimension it was meant
to improve, what any quick probe suggested, and what a later worker should try
next. If you mostly kept an inherited structure, say which one and why.

Then run the supplied field-interface check from the workspace root:

```bash
(cd solver && "$ASYS_PYTHON" -B - <<'PY'
from pathlib import Path
import torch
import yaml
from task import load_model
from check import check_field

torch.set_default_dtype(torch.float64)
config = yaml.safe_load(Path("config.yaml").read_text())
torch.manual_seed(int(config.get("seed", 0)))
print(check_field(load_model(Path("model.py"))))
PY
)
```

This checks finite derivatives and pointwise/batch consistency without fitting
or official scoring. Fix any failures, summarize your construction and stop.
