# Single-session coding controls

These are all six outputs in the manuscript's single-session comparison, with
their evaluated source, configuration and original fitted state. Each task has
three independent coding sessions. The controls share the supplied mathematical
starting information, but do not use a population of programs and guidance.

| Attempt | gCLM profile error | Fujita trajectory error |
| --- | ---: | ---: |
| 1 | 0.0007697% | 2.096% |
| 2 | 0.0007827% | 0.2215% |
| 3 | 0.001894% | 2.092% |

From the repository root, using the assessment references prepared for the main
constructions:

```bash
uv run python experiments/single_session/evaluate.py --task gclm --reference references/gclm --output /tmp/gclm-single-session.json
uv run python experiments/single_session/evaluate.py --task fujita --reference references/fujita/fujita-forward.npz --output /tmp/fujita-single-session.json
```

gCLM uses the fine integral reference on 257 logarithmic radii; Fujita uses the
same `dr dt` trajectory norm as the main comparison. Each directory's
`expected.json` gives its saved values and state hash. gCLM's effective fitting
seed is `2`, recorded in `model_config.json`; its submitted `config.json` starts
at seed `0`. The original fitted normalization is restored with the state.
These commands assess the returned programs without new coding sessions or fits.
