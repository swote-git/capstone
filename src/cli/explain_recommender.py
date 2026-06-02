#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .runtime_config import parse_args_with_config
from explainer.llm_renderer import OpenAILLMRenderer
from common.config import RecommenderConfig
from recommender.engine import ThinFilerRecommender
from common.pipeline import to_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate grounded explanations for top-K recommendations")
    p.add_argument("--data-root", type=Path, default=Path("data"), help="Dataset root directory")
    p.add_argument("--fit", action="store_true", help="Train ranker before explanation")
    p.add_argument("--max-train-users", type=int, default=800)
    p.add_argument("--sample-users", type=int, default=100)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--family", choices=["all", "deposit", "fund"], default="all")
    p.add_argument("--use-moe-harness", action="store_true", help="Use MoE harness for scoring during ranking/explanation")
    p.add_argument("--moe-debug", action="store_true", help="Enable MoE debug metadata")
    p.add_argument("--moe-ranker-weight", type=float, default=0.60)
    p.add_argument("--moe-baseline-weight", type=float, default=0.25)
    p.add_argument("--moe-utility-weight", type=float, default=0.15)
    p.add_argument("--moe-deposit-baseline-boost", type=float, default=0.05)
    p.add_argument("--moe-fund-utility-boost", type=float, default=0.10)
    p.add_argument("--moe-low-risk-fund-penalty", type=float, default=0.15)
    p.add_argument("--as-of-dates", nargs="*", default=None)
    p.add_argument("--use-llm-renderer", action="store_true", help="Use OpenAI API renderer for explanation text")
    p.add_argument("--llm-model", type=str, default="gpt-5-mini")
    p.add_argument(
        "--llm-prompt-path",
        type=Path,
        default=Path("src/explainer/explain.txt"),
        help="Path to LLM system prompt text file",
    )
    p.add_argument(
        "--no-template-fallback",
        action="store_true",
        help="Do not fallback to deterministic template when LLM output fails verifier",
    )
    p.add_argument(
        "--use-explainer-moe",
        action="store_true",
        help="Use explanation-layer MoE orchestration (reason/compliance/template experts)",
    )
    p.add_argument(
        "--explainer-moe-debug",
        action="store_true",
        help="Include candidate-level MoE debug metadata in llm_intermediate",
    )
    p.add_argument(
        "--compliance-rules-path",
        type=Path,
        default=Path("src/explainer/compliance_rules.txt"),
        help="Text file path for external compliance rules (금융소비자보호법 문항 등)",
    )
    return parse_args_with_config(p, section="explain_recommender")


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
    rec = ThinFilerRecommender(cfg)

    snapshots = rec.build_user_snapshots(as_of_dates=args.as_of_dates, sample_users=args.sample_users)
    rec.load_products()

    if args.fit:
        rec.fit(snapshots=snapshots, max_users=args.max_train_users)

    llm_renderer = None
    if args.use_llm_renderer:
        llm_renderer = OpenAILLMRenderer(model=args.llm_model, prompt_path=args.llm_prompt_path)

    result = rec.explain_recommendation_with(
        snapshots.iloc[0],
        k=args.top_k,
        llm_renderer=llm_renderer,
        fallback_to_template_on_verify_fail=not args.no_template_fallback,
        use_explainer_moe=bool(args.use_explainer_moe),
        compliance_rules_path=args.compliance_rules_path,
        explainer_moe_debug=bool(args.explainer_moe_debug),
    )
    print(to_json(result))


if __name__ == "__main__":
    main()
