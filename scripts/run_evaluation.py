"""Run baseline vs ReAct on evaluation_questions.csv."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from react_copilot.evaluation import run_full_eval


if __name__ == "__main__":
    run_full_eval()
