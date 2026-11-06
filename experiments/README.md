# Manuscript constructions and comparisons

These directories supply saved mathematical constructions, numerical controls
and assessment recipes for the current ASYS manuscript. They complement the
fresh search setups in `configs/eve/`. A new stochastic search is not expected
to return the same programs or sequence of revisions.

## Saved constructions

The links below open the original mathematical source and fitted states.
Each study's README explains the representation and how to assess it.

| Study | Construction files |
| --- | --- |
| [gCLM](gclm/) | [Selected H3 expression and state](gclm/selected/); [other ASYS outcomes and SIGS/FEX comparisons](gclm/cohort/) |
| [Transient Fujita](fujita/) | [Local and uniform response formulas, with fitted states](fujita/selected/) |
| [Periodic NLS](nonlinear_schrodinger/) | [Prescribed response construction](nonlinear_schrodinger/model.py) and [settings](nonlinear_schrodinger/config.json); no fitted state is needed |
| [Allen–Cahn](allen_cahn/) | [Original 14- and 12-scalar formulas and states](allen_cahn/selected/) |
| [Keller–Segel](keller_segel/) | [Gaussian, unchanged-family continuation, reweighted, L4 and L4/L8 fields and states](keller_segel/selected/) |
| [Waiting profile](waiting_profile/) | [Expression](waiting_profile/model.py) and [18-coefficient state](waiting_profile/state_dict.pt) |
| [Stationary Fujita](fujita_stationary/) | [Expression](fujita_stationary/model.py), [38-coefficient state](fujita_stationary/state_dict.pt) and [printed coefficients](fujita_stationary/coefficients.json) |
| [Graveleau](graveleau/) | [Full 24-scalar pressure construction and state](graveleau/selected/) |
| [Hamilton–Jacobi](hj_inverse/) | [Polynomial](hj_inverse/polynomial/) and [exponential](hj_inverse/exponential/) time-basis sources and fitted states |

## Comparisons and supporting studies

The `neural/` directories contain the manuscript's neural comparison baselines:
model definitions, fitting recipes and saved checkpoints. The ASYS expressions
for those studies are in `selected/`. The comparison models are assessed
alongside the saved expressions and are not inputs to new ASYS searches.

| Comparison | Included artifacts |
| --- | --- |
| [Allen–Cahn neural baselines](allen_cahn/neural/) | 12,801-parameter model and three fitted states; [assessment commands](allen_cahn/README.md) |
| [Fujita neural baselines](fujita/neural/) | 12,739-parameter model, paired initializations and dense/sparse fitted states; [assessment commands](fujita/README.md) |
| [gCLM matched coordinates](gclm/matched_coordinates/) | Four fits comparing shape families and coefficient coordinates |
| [Fujita response swaps](fujita/response_swap/) | Initial, local-fit and uniform-fit coefficients applied to both response definitions |
| [Fourth harmonic](allen_cahn/fourth_harmonic/) | All PDE/curvature cases, contours, source and refinement readout |
| [Single-session controls](single_session/) | All six fitted gCLM/Fujita outputs |
| [Waiting-profile derivation](waiting_profile/derivation/) | Exact logarithmic recurrence and high-precision checks |

## Assessment

Each study README identifies its model/state, metric, units and commands.
`expected.json` files contain saved measurements, not tolerances that a new
search must meet. Reference generators use the existing scientific solvers;
their large field banks are generated separately. These commands assess the
specified states without refitting.

Use the repository environment (`uv sync`; `uv sync --extra baseline` for the
high-precision tail derivations). Run commands from the repository root.
Outputs go to paths supplied on the command line. Check the delivered files
with `sha256sum -c experiments/SHA256SUMS`. Source origins and portability
changes are summarized in [SOURCES.md](SOURCES.md).

The saved states support assessment of the manuscript's constructions. They
remain separate from fresh worker tasks and their reusable initializations.
