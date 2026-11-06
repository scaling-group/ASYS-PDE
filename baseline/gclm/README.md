# gCLM SIGS and FEX baselines

The original cohort searches the real-line similarity profile with SIGS slots
2 and 4, and FEX depths 2 and 3, each with seeds 0, 1 and 2. The tail-capable
cohort uses its expanded expression families with SIGS slots 2 and FEX depth 2,
each with seeds 0 and 1. Both fit at the fixed parameter anchor 0.25.

| Cohort | Coarse phase allowance | Retained pool | Terminal fit per expression |
| --- | ---: | ---: | ---: |
| `original` | 18,000 s | 10 | 1,200 s / 20,000 Adam updates |
| `tail_capable` | 7,200 s | 4 | 1,200 s / 20,000 Adam updates |

The original FEX cohort additionally caps accumulated coarse fitting at 12,000 s.
Each run uses one CPU thread and can take several hours. The policy and study
files retain the sampling, fitting, support and selection settings for each
cohort. Training and selection do not load numerical reference banks.

```bash
uv sync --extra baseline
baseline/gclm/sigs_upstream/fetch_assets.sh

.venv/bin/python baseline/gclm/original/sigs/run_core.py --slots 2 --seed 0 --output runs/sigs-original-2-0
.venv/bin/python baseline/gclm/original/fex/run.py --depth 2 --seed 0 --output runs/fex-original-2-0
timeout --signal=KILL 14320s .venv/bin/python baseline/gclm/tail_capable/sigs/run_core.py --slots 2 --seed 0 --output runs/sigs-tail-2-0
.venv/bin/python baseline/gclm/tail_capable/fex/fex_workflow.py --seed 0 --output runs/fex-tail-2-0
```

Choose the other listed slots, depths and seeds to run the remaining cases.
Use a new output directory for every run. SIGS uses the pinned upstream decoder
and pretrained assets described in [sigs_upstream](sigs_upstream/README.md).

Run bounded initialization, coarse fitting, scoring and replay checks with:

```bash
.venv/bin/python -m pytest tests/test_gclm_baselines.py
```

## Field readout

Use the terminal candidate selected by the run's `public_selected` record:

```bash
.venv/bin/python baseline/gclm/readout.py --family original-sigs --candidate /path/to/terminal/saved --references references/gclm
```

`--family` accepts `original-sigs`, `tail-sigs`, `original-fex` and `tail-fex`.
For FEX, pass the terminal candidate directory directly. Relative errors are
reported as fractions.
