"""Frozen HJ source/state binding and independent reference readout."""

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/hj_inverse"


def test_hj_frozen_field_and_coefficient_replay(tmp_path):
    output = tmp_path / "hj.json"
    subprocess.run(
        [sys.executable, str(STUDY / "evaluate.py"), "--output", str(output)],
        cwd=tmp_path,
        check=True,
        timeout=60,
    )
    result = json.loads(output.read_text())
    expected = json.loads((STUDY / "expected.json").read_text())
    np.testing.assert_array_equal(result["coefficient_truth"], [0.115, 0.085, -0.43, -0.61])
    for arm, row in result["rows"].items():
        saved = expected[arm]
        assert saved["selected_state_included"]
        assert row["selected_state_sha256"] == saved["selected_state_sha256"]
        np.testing.assert_array_equal(row["fitted_coefficients"], saved["fitted_coefficients"])
        np.testing.assert_array_equal(result["times"], saved["times"])
        for key in (
            "relative_coefficient_errors",
            "mean_relative_coefficient_error",
            "per_frame_relative_l2",
        ):
            np.testing.assert_allclose(row[key], saved[key], rtol=5e-12, atol=1e-14)
        for domain, value in row["direct_field_relative_l2"].items():
            np.testing.assert_allclose(
                value, saved["direct_field_relative_l2"][domain], rtol=5e-12, atol=1e-14
            )
    metadata = json.loads((STUDY / "researcher_reference/assessment.json").read_text())
    observations = (
        ROOT / "configs/eve/application/asys-hj-inverse/problems/inverse/"
        "hj_coefficients/observations.npz"
    )
    assert hashlib.sha256(observations.read_bytes()).hexdigest() == metadata["observations_sha256"]


def test_hj_reference_recovers_anisotropic_heat_mode():
    reference = runpy.run_path(str(STUDY / "reference.py"))
    problem = {"initial_modes": [[1, 2, 0.6, 0.3]], "initial_scale": 2.0}
    times = np.array([0.0, 0.001, 0.002])
    values = reference["rollout"]([0.115, 0.085, 0.0, 0.0], problem, times, 8)
    axis = np.arange(8) * 2 * np.pi / 8
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    exact = (
        0.6
        / 8
        * np.sin(xx + 2 * yy + 0.3)[None]
        * np.exp(-(0.115 + 4 * 0.085) * times[:, None, None])
    )
    np.testing.assert_allclose(values, exact, rtol=1e-11, atol=1e-13)
