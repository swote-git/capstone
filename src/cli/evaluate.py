#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime_config import parse_args_with_config
from common.config import RecommenderConfig
from evaluate.recommender_eval import build_recommender_eval_report
from recommender.engine import ThinFilerRecommender


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate thin-file recommender (baseline vs ranker)")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--sample-users", type=int, default=1000)
    p.add_argument("--max-train-users", type=int, default=800)
    p.add_argument("--max-eval-users", type=int, default=300)
    p.add_argument("--fit", action="store_true", help="Fit LightGBMRanker before evaluation")
    p.add_argument("--family", choices=["all", "deposit", "fund"], default="all")
    p.add_argument("--use-moe-harness", action="store_true", help="Use MoE harness for evaluation-time scoring")
    p.add_argument("--moe-debug", action="store_true", help="Enable MoE debug metadata")
    p.add_argument("--moe-ranker-weight", type=float, default=0.60)
    p.add_argument("--moe-baseline-weight", type=float, default=0.25)
    p.add_argument("--moe-utility-weight", type=float, default=0.15)
    p.add_argument("--moe-deposit-baseline-boost", type=float, default=0.05)
    p.add_argument("--moe-fund-utility-boost", type=float, default=0.10)
    p.add_argument("--moe-low-risk-fund-penalty", type=float, default=0.15)
    p.add_argument("--ks", nargs="+", type=int, default=[5, 10])
    p.add_argument("--as-of-dates", nargs="*", default=None)
    return parse_args_with_config(p, section="evaluate")


def main() -> None:
    args = parse_args()
    cfg = RecommenderConfig(
        data_root=args.data_root,
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
    rec = ThinFilerRecommender(cfg)

    snapshots = rec.build_user_snapshots(
        as_of_dates=args.as_of_dates,
        sample_users=args.sample_users,
    )
    rec.load_products()

    report = build_recommender_eval_report(
        rec=rec,
        snapshots=snapshots,
        user_key=cfg.user_key_11,
        ks=args.ks,
        max_eval_users=args.max_eval_users,
    )
    train_df = report.pop("train_df")
    report.pop("eval_df", None)

    if args.fit:
        rec.fit(snapshots=train_df, max_users=args.max_train_users)
        report = build_recommender_eval_report(
            rec=rec,
            snapshots=snapshots,
            user_key=cfg.user_key_11,
            ks=args.ks,
            max_eval_users=args.max_eval_users,
        )
        report.pop("train_df", None)
        report.pop("eval_df", None)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
