"""Fit the original construction or score a state using the released task evaluator."""
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parents[3] / "configs/eve/application/asys-blowup-graveleau-pme-2d"
sys.path.insert(0, str(TASK))
from asys_task.evaluate import evaluate

if __name__ == "__main__":
    evaluate(__file__)
