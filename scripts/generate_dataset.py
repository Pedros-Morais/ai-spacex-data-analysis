#!/usr/bin/env python
"""Generates the versioned synthetic fallback dataset (>= 1,000 rows).

Use when the API is unavailable or to regenerate the example snapshot
that ships with the repository.

Usage:
    python scripts/generate_dataset.py [n_linhas]
"""

from __future__ import annotations

import logging
import sys

from launch_success.config import SETTINGS
from launch_success.data.synthetic import write_synthetic_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    """Generates the synthetic CSV; accepts the number of rows as an argument."""
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    frame = write_synthetic_dataset(n_rows=n_rows, settings=SETTINGS)
    print(f"Synthetic dataset generated: {len(frame)} rows -> {SETTINGS.processed_csv}")
    print(f"Launch success rate: {frame['success'].mean():.1%}")
    landings = frame["landing_success"].dropna()
    if not landings.empty:
        print(f"Landing success rate (among attempts): {landings.mean():.1%}")


if __name__ == "__main__":
    main()
