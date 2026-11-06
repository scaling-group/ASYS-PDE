# Earlier Allen–Cahn hard-jet neural control

This baseline accompanies the earlier geometric-field study. It uses a four-layer,
width-64 tanh correction with the initial field and initial PDE derivative imposed
by the analytic scaffold, on the periodic square through time 8.

| Setting | Collocation points | Initial points | Boundary pairs | Fit allowance |
| --- | ---: | ---: | ---: | ---: |
| `common1200` | 2,048 | 1,024 | 256 | 1,200 s |
| `dense1200` | 16,384 | 4,096 | 256 | 1,200 s |
| `dense2400` | 16,384 | 4,096 | 256 | 2,400 s |

All settings use the same model, loss, L-BFGS optimizer and fixed scoring code.
Each command fits fresh seeds 1101101, 1101102 and 1101103 on one CPU thread:

```bash
export ASYS_PYTHON="$PWD/.venv/bin/python"
baseline/allen_cahn/run.sh common1200 /path/to/output
baseline/allen_cahn/run.sh dense1200 /path/to/output
baseline/allen_cahn/run.sh dense2400 /path/to/output
```

An optional third argument selects one seed. `ASYS_NUM_THREADS` and
`ASYS_WALL_TIME` override the numerical thread count and fit allowance.
Existing run directories are rejected. Each fit exports its source, configuration,
state and training history, then scores the frozen field in a fresh interpreter.

## Field and interface readout

```bash
scripts/prepare_references.sh allen-cahn-geometric
.venv/bin/python baseline/allen_cahn/readout.py --log /path/to/evaluation --reference references/allen-cahn-geometric/n512-dt0.002.npz
```

The evaluation directory contains `source/`, `effective_config.yaml` and
`state.pt`. Relative field errors are reported as fractions.
