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
    p.add_argument("--use-moe-harness", action="store_true", help="Use MoE harness for score aggregation")
    p.add_argument("--moe-debug", action="store_true", help="Include MoE routing debug info in output")
    p.add_argument("--moe-ranker-weight", type=float, default=0.60)
    p.add_argument("--moe-baseline-weight", type=float, default=0.25)
    p.add_argument("--moe-utility-weight", type=float, default=0.15)
    p.add_argument("--moe-deposit-baseline-boost", type=float, default=0.05)
    p.add_argument("--moe-fund-utility-boost", type=float, default=0.10)
    p.add_argument("--moe-low-risk-fund-penalty", type=float, default=0.15)
    p.add_argument(
        "--as-of-dates",
        nargs="*",
        default=None,
        help='Optional quarter filter, e.g. "2022Q2 2022Q3"',
    )
    return parse_args_with_config(p, section="run_recommender")


def main() -> None:
    args = parse_args()

    cfg = RecommenderConfig(
        data_root=args.data_root,
        top_k=args.top_k,
        recommender_family=args.family,
        use_moe_harness=bool(args.use_moe_harness),
        moe_debug=bool(args.moe_debug),
        moe_default_weights={
            "ranker": float(args.moe_ranker_weight),
            "baseline": float(args.moe_baseline_weight),
            "utility": float(args.moe_utility_weight),
        },
        moe_deposit_baseline_boost=float(args.moe_deposit_baseline_boost),
        moe_fund_utility_boost=float(args.moe_fund_utility_boost),
        moe_low_risk_fund_penalty=float(args.moe_low_risk_fund_penalty),
    )
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
