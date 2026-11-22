"""Download or locate FER2013, validate it, and save normalized split arrays."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import prepare_data


def main() -> None:
    data = prepare_data(quick=False)
    print("Prepared official FER2013 splits:")
    for split, (_, labels) in data.items():
        print(f"  {split}: {len(labels):,} faces")
    print(f"Saved normalized arrays under {ROOT / 'data' / 'processed'}")


if __name__ == "__main__":
    main()
