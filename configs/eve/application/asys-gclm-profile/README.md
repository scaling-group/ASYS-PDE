# Real-line gCLM similarity profile

Construct an explicit, differentiable odd positive profile for the real-line
gCLM similarity equation at a=.25:

    f + beta*y*f_y + a*lambda*V*f_y - lambda*g*f = 0
    g = -H_R[f], V'=g, V(0)=0, lambda=(1+beta)/(1-a).

The canonical gauges are f(0)=0, f'(0)=g(0)=1, beta>0. Seek a decaying,
positive-half-line profile with finite weighted normalization moment. This is one parameter value, not a
periodic trajectory or across-a family. Known mathematical symmetries,
Hilbert identities, compactification and the leading power envelope with an
adjustable beta are supplied priors, not discoveries.

Only `model.py` is editable. It must expose `build_model(config)` returning a
CPU float64 torch.nn.Module, `forward(x)` mapping a 1-D real tensor to the same
shape, and `beta()` returning a positive finite scalar tensor. All numerical
parameters/buffers required for prediction must be registered and replayable.
Use explicit analytic expressions with named mathematical parameters. Neural
networks, trajectory tables and numerical equation solvers as candidate
representations are forbidden. No expression-slot or parameter-count cap is imposed.
Report complexity rather than inferring efficiency from parameter count.

The raw h must be odd, zero at origin with finite positive derivative k, positive
for x>0, and have finite positive mass A=(2/pi)*integral h(exp(t))dt. Protected
evaluation measures A and k once after fitting and freezes f(y)=h((A/k)*y)/A.
It does not repair a nonzero origin, negative profile, bad slope or unresolved
mass. Inference may evaluate this saved explicit representation; it may not fit,
solve the defining PDE/integral equation, normalize afresh, load reference tables
or use probe-dependent corrections. Parameterless explicit constructions have
a declared no-fit validation/export pathway and remain eligible if qualified.

Admissibility uses the finite weighted moment A=(2/pi)*integral h(x)/x dx,
not a new ordinary half-line L1 requirement. The beta=1 starter has finite A
and divergent ordinary L1 mass. Finite mesh/operator checks certify their
instrument resolution only, not global decay or full-line integrability.

Protected fitting uses 20,000 Adam updates, initial lr .01 with cosine schedule;
the last complete measured update is retained. It has a 1200-second allowance
including validation reserve 30s; the whole evaluation is capped at 1500s.
Public ranking is minus the qualified score at a=.25.

Current-history native source/guidance and public evaluation logs may inform
source revision.
Keep any initialization assumptions in model source explicit. The starter
uses the supplied unfitted envelope, beta initialized at 1, and zero correction.
No reference beta or reference fields may initialize candidates or enter fitting.

Do not edit or execute protected evaluation while acting as a source proposer.
Small local algebra/interface checks within the actor allowance are permitted;
formal fitting and judging belong to native evaluation. Do not inspect other
runs or private reference data. Live primary
literature research is available; attribute mathematical priors and reused ideas.

Finish one coherent candidate, explain its structural hypothesis, and preserve
uncertainty and adverse public diagnostics.

Numerical coefficient fitting, Hilbert-transform quadrature and normalization
are owned by the protected evaluator. Candidate construction initializes the
explicit formula; it must not solve the profile equation or fit coefficients.
