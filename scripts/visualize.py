"""Clean CLI Entrypoint for 4-Layer Multi-Camera VLA Rollout Visualization."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "04_visualize_vision_rollout.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
