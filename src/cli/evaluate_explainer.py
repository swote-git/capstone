#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime_config import parse_args_with_config
from evaluate.explainer_eval import evaluate_explainer_batch, sample_users
from explainer.service import GroundedExplainer
from explainer.llm_renderer import OpenAILLMRenderer
from common.config import RecommenderConfig
from recommender.engine import ThinFilerRecommender


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch evaluation for grounded recommendation explainer")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--sample-users", type=int, default=300)
    p.add_argument("--max-eval-users", type=int, default=80)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--fit", action="store_true", help="Train ranker before explanation")
    p.add_argument("--family", choices=["all", "deposit", "fund"], default="all")
    p.add_argument("--max-train-users", type=int, default=200)
    p.add_argument("--as-of-dates", nargs="*", default=None)
    p.add_argument("--use-llm-renderer", action="store_true")
    p.add_argument("--llm-model", type=str, default="gpt-5-mini")
    p.add_argument("--no-template-fallback", action="store_true")
    return parse_args_with_config(p, section="evaluate_explainer")


def main() -> None:
    args = parse_args()
    cfg = RecommenderConfig(data_root=args.data_root, top_k=args.top_k, recommender_family=args.family)
    rec = ThinFilerRecommender(cfg)

    snapshots = rec.build_user_snapshots(as_of_dates=args.as_of_dates, sample_users=args.sample_users)
    rec.load_products()

    if args.fit:
        rec.fit(snapshots=snapshots, max_users=args.max_train_users)

    eval_snapshots = sample_users(
        snapshots,
        user_col=cfg.user_key_11,
        max_users=args.max_eval_users,
        random_state=cfg.random_state,
    )

    llm_renderer = OpenAILLMRenderer(model=args.llm_model) if args.use_llm_renderer else None
    explainer = GroundedExplainer(
        rec,
        llm_renderer=llm_renderer,
        fallback_to_template_on_verify_fail=not args.no_template_fallback,
    )

    batch = evaluate_explainer_batch(explainer=explainer, eval_snapshots=eval_snapshots, top_k=args.top_k)

    report = {
        "config": {
            "fit": bool(args.fit),
            "family": str(args.family),
            "sample_users": int(args.sample_users),
            "max_eval_users": int(args.max_eval_users),
            "top_k": int(args.top_k),
            "use_llm_renderer": bool(args.use_llm_renderer),
            "llm_model": str(args.llm_model),
            "template_fallback": bool(not args.no_template_fallback),
        },
        "snapshot_quality": rec.snapshot_quality_report(snapshots),
        "coverage": {
            **batch["coverage"],
            "evaluated_users": int(eval_snapshots[cfg.user_key_11].nunique()),
        },
        "metrics": batch["metrics"],
        "reason_feature_distribution_top10": batch["reason_feature_distribution_top10"],
        "reason_pattern_distribution_top10": batch["reason_pattern_distribution_top10"],
        "failed_examples": batch["failed_examples"],
        "warnings": [],
    }

    if report["snapshot_quality"]["cb_join_rate"] < 0.05:
        report["warnings"].append(
            "cb_join_rate is very low; explanations likely reflect 11+12-only behavior."
        )
    if report["metrics"]["reason_pattern_diversity"] < 0.05:
        report["warnings"].append(
            "Very low reason-pattern diversity; explanation quality may be degenerate despite perfect verifier pass."
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
