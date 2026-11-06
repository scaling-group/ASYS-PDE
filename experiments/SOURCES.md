# Sources and portability

The selected model definitions and serialized states retain their recovered
bytes. Assessment entry points load those states without fitting. Printed
stationary Fujita coefficients and the standalone Graveleau component derivation
accompany their full fitted constructions. `SHA256SUMS` covers the delivered
files, including sources, configurations, measurements and states.

| Material | Source |
| --- | --- |
| gCLM H3 | [Selected expression and state](gclm/selected/), recovered from development commit `b4853df536511c4de65c4167e0a8c49cb54ae083` and independently matched to the original H3 packet |
| gCLM selected cohort | Selected-terminal archive below: five ASYS outcomes H1–H5 and SIGS-0/1 terminal-02/03, FEX-0/1 terminal-02/00; original sources, normalization and states |
| gCLM matched-coordinate fits | Manuscript collection `paper-assets/asys/research-briefs/`: original `source/`, four `results/` arms and reference definitions |
| Allen–Cahn 14/12-scalar formulas | `allen-cahn-frozen-official-20261031.zip`, candidates `solver_51e8e238b607` and `solver_ac866023528b` |
| Allen–Cahn neural controls | AC neural archive below: `frozen/`, three `results/*-nn-lbfgs/` states and the full-trajectory readout |
| Fujita local/uniform formulas and neural controls | Dense and sparse Fujita archives below: `paper300/arms/{L,U}/`, `frozen/`, paired initializations and three seeds per sampling setting |
| Fujita response swaps | [Two response definitions](fujita/selected/) and [initial/local-fit/uniform-fit states](fujita/response_swap/), recovered from development commit `b4853df536511c4de65c4167e0a8c49cb54ae083` |
| NLS prescribed response | Source/config bundle below, `nls/solver_fbffd848c079/application/`; prescribed settings and assessment definitions from the NLS/stationary Fujita supplement archive |
| Keller–Segel | Original-artifact packet below: Gaussian `solver_cfc3a9238a37`, reweighted `solver_2d869f3858ea`, Laguerre-four `solver_e4da6f5b83fa`; selected-terminal archive: unchanged-family continuation `solver_86044623685f`, L4/L8 `solver_27f90b2ff9f5` and original assessment records |
| Waiting profile | Original-artifact packet: `solver_cc8cce87c7bf`, frozen evaluator, exact recurrence and high-precision asymptotic analysis |
| Stationary Fujita | Manuscript repository commit `338ea467189200748ce768e5d9af2b3fc6788786`, `output/results-overview-sample/fujita-source.txt`; original `solver_33c01b8f547c` state and 11,001-node residual assessment from the HJ/Graveleau/Fujita archive below; original reference generator from the original-artifact packet |
| Graveleau | HJ/Graveleau/Fujita archive: iteration-37 `solver_cff4079e5ca7`, full 24-scalar source/state, original field and contraction readouts |
| Hamilton–Jacobi | Source/config bundle, paired `solver_207f33402100` and `solver_21476c1baa1a` definitions; HJ/Graveleau/Fujita archive: exact selected states, frozen readouts, reference arrays and generation recipes |
| Single-session controls | [All six evaluated applications, states and readouts](single_session/), recovered from development commit `b4853df536511c4de65c4167e0a8c49cb54ae083` |
| Fourth-harmonic experiment | [Sources, measurements and readout](allen_cahn/fourth_harmonic/), recovered from development commit `edc2e966050b7a8d0d70953f9b1e322688ab26ad` and checked against `ac-fourth-harmonic-test-20261105.zip` |

The following hashes identify the recovered archives. They are provenance
records; the release contains the scientific files needed by each study rather
than requiring these archives at runtime.

| Archive | SHA256 |
| --- | --- |
| `allen-cahn-frozen-official-20261031.zip` | `7f5aa55dc539e70a4f39b9eb036bfd2460fb02aedd07a40c1eac874019bf0247` |
| AC neural: `ac-neural-final-review/portable-evidence.zip` | `1eac452662e94eb04dbafeb55766f3a5fcf81e5436ed267467b0f2ebbacc3ac0` |
| Fujita dense: `fujita-neural-density-review/portable-evidence.zip` | `2c6173c147a5a3e0d36341bc2d25a64cd1eff1ca888e9d7f33f42acaae447f56` |
| Fujita sparse: `fujita-neural-final-review/portable-evidence.zip` | `2c73afd4f4503b19814a34504a1d3b4596d02f6d3549a788b08d588df7bff08f` |
| `methods-recovered-20261104/source-config-bundle.zip` | `450163a66236ac324e4be6f3428b5716426e4419ebfda94bc98999763d3a2ec4` |
| `nls-stationary-fujita-supplement-evidence-20261029.zip` | `62e5a984cd73103a8249c80ad6f0e3a43af6cfaa01206b9fa1a15f2e090cd00f` |
| `asys-local-originals-20261106.zip` | `a7447fa603d67c6f2f96510667fb92d4d96dbb1bcf10cbfb0ea855ba66a993b9` |
| `ac-fourth-harmonic-test-20261105.zip` | `f0647eb9a2f45e540f908cd490a146142c90192360478409877d35d984900189` |
| `HJ-Graveleau-stationary-Fujita-originals.zip` | `1b14dbf5850fc70ae9d2b7a7b5349fc880a8c79a205b50ab247046237240f02c` |
| `gCLM-selected-terminals-KS-controls-originals.zip` | `c5c8277fcafadbd2fe2277f7c91d95075a9084e2a7a6d49ee025ba3c6ac63277` |

Portable entry points use relative source paths and explicit output paths.
Scheduler requirements and machine-specific launch records have been removed.
The neural fitters retain their original objectives, sampling seeds, optimizer
schedules, completed-step selection, rollback and time accounting. Dense and
sparse Fujita sampling share the original generator and paired initial states;
checks use the sample geometry and state values rather than a platform-sensitive
hash of floating-point coordinates.

Assessment code retains each study's norm, grid and reference convention.
NumPy integration calls use `trapezoid`. Stationary Fujita's portable residual
diagnostic independently differentiates the recovered expression on its original
11,001-node residual mesh; its reference-profile comparison uses a separate
14,001-node grid. The Hamilton–Jacobi reference array belongs to researcher
assessment and is separate from worker observations. Large reference banks and
complete diffuse-field snapshots are generated separately using the supplied
scientific solvers.
