#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

from scripts.common.runtime_config import parse_args_with_config
from thin_filer.explainer import GroundedExplainer
from thin_filer.llm_renderer import OpenAILLMRenderer
from thin_filer.pipeline_config import RecommenderConfig
from thin_filer.recommender import ThinFilerRecommender


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


def _sample_users(df, user_col: str, max_users: int, random_state: int):
    users = df[user_col].drop_duplicates()
    if len(users) <= max_users:
        return df
    selected = users.sample(n=max_users, random_state=random_state)
    return df[df[user_col].isin(selected)].copy()


def main() -> None:
    args = parse_args()
    cfg = RecommenderConfig(data_root=args.data_root, top_k=args.top_k, recommender_family=args.family)
    rec = ThinFilerRecommender(cfg)

    snapshots = rec.build_user_snapshots(as_of_dates=args.as_of_dates, sample_users=args.sample_users)
    rec.load_products()

    if args.fit:
        rec.fit(snapshots=snapshots, max_users=args.max_train_users)

    eval_snapshots = _sample_users(
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

    alignments: List[float] = []
    hallucinations: List[float] = []
    fact_consistency: List[float] = []
    pass_flags: List[float] = []
    forbidden_flags: List[float] = []
    reasons_per_item: List[int] = []
    reason_feature_counts: Dict[str, int] = {}
    reason_patterns: Dict[str, int] = {}
    failed_examples: List[Dict[str, object]] = []

    for _, user_row in eval_snapshots.iterrows():
        out = explainer.explain_top_k(user_row, k=args.top_k)
        for rec_item in out["recommendations"]:
            ver = rec_item["verification"]
            alignments.append(float(ver["reason_alignment"]))
            hallucinations.append(float(ver["hallucination_rate"]))
            fact_consistency.append(1.0 if bool(ver["fact_consistency"]) else 0.0)
            pass_flags.append(1.0 if bool(ver["passed"]) else 0.0)
            forbidden_flags.append(1.0 if len(ver["forbidden_claims"]) > 0 else 0.0)

            reason_signals = rec_item.get("reason_signals", [])
            reasons_per_item.append(len(reason_signals))
            for s in reason_signals:
                f = str(s.get("feature", "unknown"))
                reason_feature_counts[f] = reason_feature_counts.get(f, 0) + 1
            pat = "|".join(sorted(str(s.get("feature", "unknown")) for s in reason_signals))
            reason_patterns[pat] = reason_patterns.get(pat, 0) + 1

            if not bool(ver["passed"]) and len(failed_examples) < 5:
                failed_examples.append(
                    {
                        "user_id": out["user_id"],
                        "product_id": rec_item["product_id"],
                        "verification": ver,
                        "rendered_explanation": rec_item["rendered_explanation"],
                    }
                )

    def mean_or_zero(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    top_reason_features = sorted(reason_feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]

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
            "evaluated_users": int(eval_snapshots[cfg.user_key_11].nunique()),
            "evaluated_explanations": int(len(pass_flags)),
        },
        "metrics": {
            "reason_coverage_rc": mean_or_zero(alignments),
            "hallucination_rate_hr": mean_or_zero(hallucinations),
            "fact_consistency_rate": mean_or_zero(fact_consistency),
            "verification_pass_rate": mean_or_zero(pass_flags),
            "forbidden_claim_rate": mean_or_zero(forbidden_flags),
            "avg_reasons_per_item": mean_or_zero([float(x) for x in reasons_per_item]),
            "reason_pattern_diversity": float(len(reason_patterns) / max(1, len(pass_flags))),
        },
        "reason_feature_distribution_top10": [
            {"feature": feat, "count": int(cnt)} for feat, cnt in top_reason_features
        ],
        "reason_pattern_distribution_top10": [
            {"pattern": p, "count": int(c)}
            for p, c in sorted(reason_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        ],
        "failed_examples": failed_examples,
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
