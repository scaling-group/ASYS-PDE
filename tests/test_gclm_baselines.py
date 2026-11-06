"""Baseline cohort settings and bounded real numerical initialization checks."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1] / "baseline/gclm"


@pytest.mark.parametrize(
    "cohort,slots,depths,seeds,pool",
    [
        ("original", [2, 4], [2, 3], [0, 1, 2], 10),
        ("tail_capable", [2], [2], [0, 1], 4),
    ],
)
def test_cohort_settings(cohort, slots, depths, seeds, pool):
    sigs = json.loads((ROOT / cohort / "sigs/policy.json").read_text())
    fex = yaml.safe_load((ROOT / cohort / "fex/study.yaml").read_text())
    assert sigs["proposed_search"]["slots"] == slots
    assert sigs["proposed_search"]["seeds"] == fex["seeds"] == seeds
    assert fex["depths"] == depths
    assert sigs["proposed_search"]["pool"] == fex["pool_size"] == pool
    assert sigs["terminal"]["stages"] == fex["terminal"]["stages"]
    for path in (ROOT / cohort).rglob("*"):
        if path.is_symlink():
            assert path.resolve().is_relative_to(ROOT)
            assert path.is_file()


@pytest.mark.integration
@pytest.mark.parametrize("cohort", ["original", "tail_capable"])
@pytest.mark.parametrize("method", ["sigs", "fex"])
def test_real_coarse_fit(cohort, method, tmp_path):
    upstream = Path(os.environ.get("ASYS_SIGS_UPSTREAM", ROOT / "sigs_upstream"))
    if method == "sigs" and not (upstream / "assets/model.ckpt").is_file():
        pytest.skip("Download the pinned SIGS assets before numerical initialization checks")
    command = r"""
import json, math, sys, time
from pathlib import Path
import torch, yaml
source, output, upstream = map(Path, sys.argv[1:])
sys.path.insert(0, str(source))
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
if source.name == 'sigs':
    sys.path.insert(0, str(upstream/'upstream/src'))
    from search_core import Engine
    from structure import Proposals
    policy = json.loads((source/'policy.json').read_text())
    engine = Engine(upstream, policy)
    sampler = Proposals(engine.atoms, 2, 910200, 5)
    proposal = next(sampler.sample(sampler.broad(), 1))
    result = engine.fit(output, 'coarse-00000', proposal, 'coarse', time.monotonic()+60)
    assert result['fit']['updates'] > 0, result
    assert (output/'coarse-00000/state_dict.pt').is_file(), result
    assert result['fit_eligible'], result
    # Exercise the released decoder on the actual proposal's latent coordinates.
    assert len(engine.decoder(proposal['latent'], 2)) == 2
    from structure import candidate
    from single_anchor import FrozenProfile, save_profile
    raw = candidate(proposal)
    state = torch.load(output/'coarse-00000/state_dict.pt', weights_only=True)
    raw.load_state_dict(state, strict=True)
    frozen = FrozenProfile(raw, result['fit']['validation']['mass'])
    save_profile(frozen, output/'coarse-00000/saved')
    public = engine.public(output, result, time.monotonic()+120)
    assert public['assessment']['status'] == 'qualified', public
    assert public['replay_ok'] and public['saved_bytes_unchanged'], public
    print(json.dumps({'fit_eligible':result['fit_eligible'], 'loss':result['loss'],
                      'public_score':public['assessment']['score'],
                      'replay_ok':public['replay_ok']}))
else:
    settings = yaml.safe_load((source/'study.yaml').read_text())
    if source.parent.name == 'original':
        from run import Numerical
        engine = Numerical(source, settings)
    else:
        from fex_workflow import CommonNumerical
        engine = CommonNumerical(settings)
    policy, optimizer = engine.controller(2, 0)
    actions = engine.sample(policy, 2)
    config = dict(depth=2, actions=actions[0], seed=1200000,
                  initial_beta=settings['initial_beta'], initial_sd=settings['initial_sd'])
    result = engine.fit(output/'candidate', config, settings['coarse']['stages'],
                        settings['coarse']['fit_seconds_per_proposal'])
    assert result['completed'], result
    assert (output/'candidate/state_dict.pt').is_file()
    score = engine.score(output/'candidate')
    assert score['admissible'] and math.isfinite(score['score']), score
    loss = engine.update(policy, optimizer, actions,
                         [engine.reward(result['selection_loss']), engine.reward(None)])
    assert math.isfinite(loss)
    print(json.dumps({'score':score['score'], 'controller_loss':loss}))
"""
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", command, str(ROOT / cohort / method), str(tmp_path), str(upstream)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
