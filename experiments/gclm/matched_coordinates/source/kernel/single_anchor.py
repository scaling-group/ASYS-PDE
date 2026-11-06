"""Single-parameter profile adapter with frozen normalization.

The released expression/profile semantics are reused unchanged. This module is
not an across-a family and does not import numerical reference data.
"""
import copy
import math
import json
from pathlib import Path
import torch

ANCHOR = 0.25


class FrozenProfile(torch.nn.Module):
    def __init__(self, raw, center_mass):
        super().__init__()
        if not math.isfinite(float(center_mass)) or float(center_mass) <= 0:
            raise ValueError("Positive finite frozen center mass required")
        self.raw = copy.deepcopy(raw)
        self.raw.requires_grad_(False)
        self.register_buffer("center_mass", torch.tensor(float(center_mass), dtype=torch.float64))

    def _anchor(self, a):
        a = torch.as_tensor(a, dtype=torch.float64)
        if not bool(torch.isfinite(a).all() and (a == ANCHOR).all()):
            raise ValueError("This saved object declares only a=0.25")
        return a

    def forward(self, a, y):
        a = self._anchor(a)
        y = torch.as_tensor(y, dtype=torch.float64)
        _, y = torch.broadcast_tensors(a, y)
        return self.raw(self.center_mass * y) / self.center_mass

    def beta(self, a):
        a = self._anchor(a)
        return self.raw.beta().expand(a.shape)


def choose_terminal(rows):
    """Common positive loss only; earliest terminal ordinal wins exact ties."""
    candidates = []
    for row in rows:
        a = row.get("assessment", {})
        loss = a.get("score")
        if (row.get("fit_eligible") and a.get("status") == "qualified"
                and isinstance(loss, (int, float)) and not isinstance(loss, bool)
                and math.isfinite(loss) and loss >= 0):
            candidates.append((loss, row["ordinal"], row["identity"]))
    return min(candidates)[2] if candidates else None


def save_profile(frozen, path):
    """Create an immutable-state package; no fit or judge inside export."""
    from expression import Profile
    if not isinstance(frozen.raw, Profile):
        raise TypeError("SIGS raw Profile required")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    metadata = dict(expression=frozen.raw.expression.original, anchor=ANCHOR,
                    center_mass=float(frozen.center_mass), initial_beta=1.0,
                    normalization="creation-time frozen state")
    (path/'candidate.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    torch.save(frozen.state_dict(),path/'state_dict.pt')


def load_profile(path):
    """Pure saved-state reconstruction; no normalization is recomputed."""
    from expression import Profile
    path = Path(path)
    metadata = json.loads((path/'candidate.json').read_text())
    if metadata['anchor'] != ANCHOR:
        raise ValueError("Unexpected saved task declaration")
    raw = Profile(metadata['expression'], beta=metadata['initial_beta'])
    frozen = FrozenProfile(raw,metadata['center_mass'])
    frozen.load_state_dict(torch.load(path/'state_dict.pt',weights_only=True,map_location='cpu'),strict=True)
    if float(frozen.center_mass) != metadata['center_mass']:
        raise ValueError("Saved normalization metadata/state disagreement")
    return frozen
