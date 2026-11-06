# Mathematical starting hypotheses

Explore an inspectable
nonnegative core/halo density, a smooth cumulative radial mass differentiated
into density, or an analytic correction to the Gaussian initial density. The
representation, loss and fitting method are your scientific choices.

The public stationary critical-core formula 8*a/(a+r^2)^2, a>0, has radial
mass 4. It does not supply the Gaussian IVP, its halo, time law or finite-window
solution. Disclose this prior if used. A chosen clock or exact initial anchor
alone does not establish useful later dynamics. Do not infer solution fidelity
from a small force-normalized residual alone.

Raw public scoring does not require raw training loss. Force normalization,
coordinate scaling, least-squares curvature or other justified conditioning
may help fitting; they can also suppress important dynamics. Check gradients,
initialization sensitivity, early/late temporal coverage and signed residuals.
If a recipe fails, inspect the inherited loss history and effective number of
steps before attributing the failure to class capacity. Match mathematical
objectives when claiming a faster implementation. Small synthetic timings are
feasibility evidence, not full-fit or query-latency benchmarks.

Use derivative/batch checks, fitting history and the task's public feedback to
choose a useful next hypothesis. Count fixed numerical expansions/buffers as well
as trainable scalars. The numerical evaluator, not the worker, performs fitting.
