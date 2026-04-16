#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import RecommenderConfig
from recommender.engine import ThinFilerRecommender


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit 11↔09 join integrity for lagged snapshot build")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--sample-users", type=int, default=5000)
    p.add_argument("--sample-size", type=int, default=10000, help="Rows sampled for key-profile analysis")
    p.add_argument("--as-of-dates", nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RecommenderConfig(data_root=args.data_root)
    rec = ThinFilerRecommender(cfg)

    snapshots = rec.build_user_snapshots(
        as_of_dates=args.as_of_dates,
        sample_users=args.sample_users,
    )
    report = rec.join_diagnostics(snapshots=snapshots, sample_size=args.sample_size)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
