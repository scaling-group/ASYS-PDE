"""Frozen-state comparisons use the paper's samples and reference conventions."""

import subprocess
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def test_fujita_response_residuals_match_saved_diagnostics_without_changing_states():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
from pathlib import Path
import numpy as np
import torch
from evaluate import formula, residuals

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
expected = json.loads(Path('response_swap/expected.json').read_text())
for row in expected['rows']:
    model = formula(row['response'], Path('response_swap') / (row['state'] + '.pt'))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    actual = residuals(model)
    for key, value in actual.items():
        np.testing.assert_allclose(value, row[key], rtol=5e-10, atol=1e-17, err_msg=key)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
""",
        ],
        cwd=EXPERIMENTS / "fujita",
        check=True,
    )


def test_allen_cahn_keeps_original_and_common_reference_comparisons():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import evaluate as readout

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
axis = -1 + 2*np.arange(384)/384
times = np.unique(np.r_[0, 2.**np.arange(-10, -5), np.arange(1, 257)/32])
# Use one synthetic constant reference for both conventions. Broadcast one actual
# frozen-model query per time so this routing check avoids full neural trajectories.
truth = np.broadcast_to(np.ones((1, 1, 1)), (262, 384, 384))
bank = dict(x=axis, t=times, u=truth)
def predict(model, axis, time):
    with torch.inference_mode():
        value = model(torch.tensor([[.03, .01, time]])).item()
    return np.broadcast_to(value, (len(axis), len(axis)))
with tempfile.TemporaryDirectory() as directory:
    original, current = [Path(directory)/name for name in ('original.npz', 'current.npz')]
    original.with_suffix('.json').write_text(json.dumps(dict(initial_regularizer=1e-14)))
    current.with_suffix('.json').write_text(json.dumps(dict(initial_regularizer=1e-12)))
    with patch.object(readout.np, 'load', side_effect=lambda *a, **k: nullcontext(bank)), \
         patch.object(readout, 'predict', predict):
        rows = readout.evaluate(original, current)['rows']
by_key = {(row['candidate'], row['initial_regularizer']): row for row in rows}
assert len(rows) == 6
assert set(by_key) == {('formula14', 1e-14), ('formula12', 1e-14),
                      ('formula14', 1e-12), ('2026110401', 1e-12),
                      ('2026110411', 1e-12), ('2026110421', 1e-12)}
a, b = by_key['formula14', 1e-14], by_key['formula14', 1e-12]
assert a['per_time'] == b['per_time']
assert a['field_error_percent'] == b['field_error_percent']
assert a['weighted_R'] == 32.
""",
        ],
        cwd=EXPERIMENTS / "allen_cahn",
        check=True,
    )
