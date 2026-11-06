from pathlib import Path
import json, csv, math
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description="Read saved fourth-harmonic amplitudes and convergence checks."
)
parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
R = args.input
O = args.output
O.mkdir(parents=True, exist_ok=False)


def csvrows(p):
    return [{k: (float(v) if v else None) for k, v in r.items()} for r in csv.DictReader(p.open())]


def maxdiff(a, b, k):
    assert len(a) == len(b)
    return max(abs(x[k] - y[k]) for x, y in zip(a, b))


def check_contours(path, rows):
    with np.load(path, allow_pickle=False) as data:
        radii, theta = data["r"], data["theta"]
    if radii.shape != (len(rows), len(theta)) or not np.isfinite(radii).all():
        raise ValueError(f"Invalid contour record: {path}")
    mean = radii.mean(axis=1)
    for order in (2, 4):
        measured = 2 * (radii * np.cos(order * theta)).mean(axis=1) / mean
        np.testing.assert_allclose(measured, [r[f"e{order}"] for r in rows], rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(
        np.pi * (radii * radii).mean(axis=1), [r["area"] for r in rows], rtol=2e-12, atol=2e-14
    )


primary = []
allrows = {}
for p in sorted((R / "measurements").iterdir()):
    if not p.is_dir():
        continue
    d = json.loads((p / "RESULT.json").read_text())
    assert d["status"] == "complete"
    rr = csvrows(p / "metrics.csv")
    assert len(rr) == 61
    assert all((v is None or math.isfinite(v)) for r in rr for v in r.values())
    allrows[p.name] = rr
    check_contours(p / "contours.npz", rr)
    if "refinement" in p.name:
        continue
    e = rr[-1]
    a = d["a"]
    ep = d["eps"]
    primary.append(
        dict(
            case=p.name,
            epsilon=ep,
            a=a,
            initial_e4=rr[0]["e4"],
            max_spurious_or_measured_e4=max(abs(r["e4"]) for r in rr),
            final_e2=e["e2"],
            final_e4=e["e4"],
            final_prediction=e["predicted_e4"],
            final_absolute_error=e["e4_absolute_error"],
            final_relative_error_percent=(
                100 * e["e4_absolute_error"] / abs(e["predicted_e4"]) if a else None
            ),
            max_absolute_error=max(r["e4_absolute_error"] for r in rr),
            final_area=e["area"],
            final_R_area=e["R_area"],
            final_s_measured_area=e["s_measured_area"],
            final_area_diagnostic_absolute_error=e["e4_measured_area_absolute_error"],
            final_e4_over_a2=(e["e4"] / a**2 if a else None),
        )
    )
conv = {}
base = allrows["eps02-a08"]
for n in ["eps02-a08-space-refinement", "eps02-a08-time-refinement"]:
    rr = allrows[n]
    conv[n] = {k: maxdiff(base, rr, k) for k in ["e2", "e4", "r0", "area", "R_area"]}
    conv[n]["final_e4"] = rr[-1]["e4"]
    conv[n]["relative_to_smallest_predicted_endpoint_signal"] = conv[n]["e4"] / (
        (11 / 20) * 0.04**2 * (0.7**2 - 0.7**7)
    )
d = json.loads((R / "measurements/eps02-a08/RESULT.json").read_text())
conv["extraction"] = {
    "max_angular2048_to4096_e4_difference": max(
        abs(x["angular2048_to4096_e4_difference"]) for x in d["extraction_checks"]
    ),
    "max_cubic_to_quintic_e4_difference": max(
        abs(x["quintic_minus_cubic_e4"]) for x in d["extraction_checks"]
    ),
    "distance_seed_refinement_Linf_near_interface": d[
        "distance_seed_refinement_Linf_near_interface"
    ],
}
b1 = csvrows(R / "curvature-bridge/a0.08-n128.csv")
b2 = csvrows(R / "curvature-bridge/a0.08-n256.csv")
conv["bridge"] = {k: maxdiff(b1, b2, k) for k in ["e2", "e4", "r0", "area"]}
amp = {}
for eps in [0.04, 0.02]:
    a = next(x for x in primary if x["epsilon"] == eps and x["a"] == 0.04)
    b = next(x for x in primary if x["epsilon"] == eps and x["a"] == 0.08)
    amp[str(eps)] = {
        "endpoint_e4_ratio_double_amplitude": b["final_e4"] / a["final_e4"],
        "quadratic_ratio": 4.0,
        "endpoint_normalized_ratio": (b["final_e4"] / 0.08**2) / (a["final_e4"] / 0.04**2),
    }
out = dict(primary=primary, convergence=conv, amplitude_scaling=amp)
(O / "results.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
print(json.dumps(out, indent=2))
