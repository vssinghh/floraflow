"""Clean CLI Entrypoint for Multi-Camera Vision Policy Training."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "02b_train_vision_policy.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
