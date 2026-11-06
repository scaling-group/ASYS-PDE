"""Reconstruct the nine equation-selected outcomes and assess frozen profiles."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
TASK = REPO / "configs/eve/application/asys-gclm-profile"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def manifest():
    return json.loads((ROOT / "cohort/manifest.json").read_text())


def load(row):
    """Restore original coefficients and creation-time normalization only."""
    folder = ROOT / row["folder"]
    if row["method"] == "SIGS":
        loader = module(REPO / "baseline/gclm/tail_capable/sigs/single_anchor.py",
                        "gclm_cohort_sigs_loader")
        previous = sys.modules.get("expression")
        try:
            module(ROOT / "cohort/source/sigs_expression.py", "expression")
            return loader.load_profile(folder).eval()
        finally:
            if previous is None:
                sys.modules.pop("expression", None)
            else:
                sys.modules["expression"] = previous
    config = json.loads((folder / "model_config.json").read_text())
    normal = json.loads((folder / "normalization.json").read_text())
    if row["method"] == "ASYS":
        raw = module(folder / "model.py", "gclm_cohort_" + row["id"]).build_model(config)
        state = folder / "state_dict.pt"
    else:
        raw = module(ROOT / "cohort/source/fex_model.py", "gclm_cohort_fex").Model(config)
        state = folder / "frozen_state_dict.pt"
    frozen = module(ROOT / "selected/frozen_profile.py", "gclm_cohort_frozen")
    model = frozen.FrozenProfile(raw.cpu().double(), normal["amplitude"], normal["origin_slope"])
    model.load_state_dict(torch.load(state, map_location="cpu", weights_only=True), strict=True)
    for key, saved in (("amplitude", "amplitude"), ("coordinate_scale", "coordinate_scale"),
                       ("creation_slope", "origin_slope")):
        if float(getattr(model, key)) != normal[saved]:
            raise ValueError("Saved normalization metadata/state disagreement: " + row["id"])
    return model.eval()


def aggregate(rows):
    """Equal-weight medians over independently selected outcomes, by method."""
    result = {}
    for method in ("ASYS", "SIGS", "FEX"):
        selected = [row for row in rows if row["method"] == method]
        references = {}
        for bank in selected[0]["references"]:
            errors = [row["references"][bank]["profile_error_percent"] for row in selected]
            references[bank] = dict(best_profile_error_percent=min(errors),
                                    median_profile_error_percent=float(np.median(errors)),
                                    worst_profile_error_percent=max(errors))
        result[method] = dict(outcomes=len(selected), references=references)
    return result


def evaluate(reference_root, public=False):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    declaration = manifest()
    models = [(row, load(row)) for row in declaration["rows"]]
    sys.path.insert(0, str(REPO / "scripts/references"))
    from fixed_point_reference import CanonicalProfile, LogProfile
    errors = module(REPO / "baseline/gclm/readout.py", "gclm_baseline_readout").errors
    y = np.logspace(-3, 3, 257)
    truths, hashes = {}, {}
    for bank in ("coarse", "medium", "fine", "extended_domain"):
        path = Path(reference_root) / bank / "profile.npz"
        hashes[bank] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            profile = LogProfile(data["w"], data["log_values"], float(data["tail_power"]))
            truth = CanonicalProfile(profile, float(data["b"])).evaluate(y)[0]
            truths[bank] = truth, float(data["beta"])
    if public:
        assessor = module(TASK / "kernel/common_single.py", "gclm_cohort_assessor")
        assessment_config = json.loads((TASK / "assessment_config.json").read_text())
    rows = []
    for row, model in models:
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            values = model(.25, torch.tensor(y)).numpy()
            beta = float(model.beta(.25))
        if values.shape != y.shape or not np.isfinite(values).all() or not np.isfinite(beta):
            raise ValueError("Invalid frozen profile: " + row["id"])
        references = {}
        for bank, (truth, reference_beta) in truths.items():
            error = errors(values, truth, y)
            references[bank] = dict(
                profile_error_percent=100*error["log_grid_relative_l2"],
                physical_radius_l2_error_percent=100*error["physical_radius_relative_l2"],
                peak_normalized_max_error_percent=100*error["peak_normalized_maximum_error"],
                absolute_beta_error=abs(beta-reference_beta))
        result = dict(id=row["id"], method=row["method"], beta=beta,
                      parameters=sum(p.numel() for p in model.parameters()),
                      buffers=sum(p.numel() for p in model.buffers()),
                      profile=values.tolist(), references=references)
        if public:
            result["common_public_assessment"] = assessor.assess_family(
                model, anchors=(.25,), config=assessment_config)
        if any(not torch.equal(value, model.state_dict()[key]) for key, value in state.items()):
            raise ValueError("Frozen state changed during assessment: " + row["id"])
        rows.append(result)
    return dict(units="Profile errors are percentages; beta errors are absolute.",
                normalization_recomputed=False, y=y.tolist(), rows=rows,
                aggregates=aggregate(rows), reference_bank_sha256=hashes)
