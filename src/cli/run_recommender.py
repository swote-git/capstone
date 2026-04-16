#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .runtime_config import parse_args_with_config
from common.config import RecommenderConfig
from recommender.engine import ThinFilerRecommender
from common.pipeline import to_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thin-file financial recommender (offline ranking)")
    p.add_argument("--data-root", type=Path, default=Path("data"), help="Dataset root directory")
    p.add_argument("--model-path", type=Path, default=Path("artifacts/lgbm_ranker.pkl"))
    p.add_argument("--fit", action="store_true", help="Train LightGBMRanker")
    p.add_argument("--max-train-users", type=int, default=1000)
    p.add_argument("--sample-users", type=int, default=100, help="Users for recommendation demo")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--family", choices=["all", "deposit", "fund"], default="all")
    p.add_argument(
        "--as-of-dates",
        nargs="*",
        default=None,
        help='Optional quarter filter, e.g. "2022Q2 2022Q3"',
    )
    return parse_args_with_config(p, section="run_recommender")


def main() -> None:
    args = parse_args()

    cfg = RecommenderConfig(data_root=args.data_root, top_k=args.top_k, recommender_family=args.family)
    recommender = ThinFilerRecommender(cfg)

    snapshots = recommender.build_user_snapshots(
        as_of_dates=args.as_of_dates,
        sample_users=args.sample_users,
    )
    recommender.load_products()

    if args.fit:
        recommender.fit(snapshots=snapshots, max_users=args.max_train_users)
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        recommender.save(args.model_path)

    example = recommender.recommend(snapshots.iloc[0], k=args.top_k)
    print(to_json(example))


if __name__ == "__main__":
    main()
