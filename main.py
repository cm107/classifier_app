"""
main.py — Application entry point.

Responsibilities (implemented in Milestone 4):
    - Resolve the storage/ directory relative to this file.
    - Enable High-DPI scaling.
    - Instantiate QApplication and MainWindow.
    - Create storage subdirectories on first run.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Storage root — always resolved relative to main.py so the app is portable.
# ---------------------------------------------------------------------------
STORAGE_ROOT = Path(__file__).parent / "storage"


def _ensure_storage_dirs():
    for subdir in ("datasets", "models", "logs", "trash"):
        (STORAGE_ROOT / subdir).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    _ensure_storage_dirs()

    # GUI bootstrap is implemented in Milestone 4.
    # Placeholder: verify storage dirs exist.
    print(f"Storage root: {STORAGE_ROOT.resolve()}")
    for subdir in STORAGE_ROOT.iterdir():
        print(f"  {subdir.name}/")
