#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime_config import parse_args_with_config
from evaluate.explainer_eval import evaluate_explainer_batch, sample_users
from evaluate.explainer_understanding_eval import (
    ExplainerUnderstandingEvaluator,
    export_jsonl,
)
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
    p.add_argument("--use-moe-harness", action="store_true")
    p.add_argument("--moe-debug", action="store_true")
    p.add_argument("--moe-ranker-weight", type=float, default=0.60)
    p.add_argument("--moe-baseline-weight", type=float, default=0.25)
    p.add_argument("--moe-utility-weight", type=float, default=0.15)
    p.add_argument("--moe-deposit-baseline-boost", type=float, default=0.05)
    p.add_argument("--moe-fund-utility-boost", type=float, default=0.10)
    p.add_argument("--moe-low-risk-fund-penalty", type=float, default=0.15)
    p.add_argument("--max-train-users", type=int, default=200)
    p.add_argument("--as-of-dates", nargs="*", default=None)
    p.add_argument("--use-llm-renderer", action="store_true")
    p.add_argument("--llm-model", type=str, default="gpt-5-mini")
    p.add_argument(
        "--llm-prompt-path",
        type=Path,
        default=Path("src/explainer/explain.txt"),
        help="Path to LLM system prompt text file",
    )
    p.add_argument("--no-template-fallback", action="store_true")
    p.add_argument("--use-explainer-moe", action="store_true")
    p.add_argument("--explainer-moe-debug", action="store_true")
    p.add_argument(
        "--compliance-rules-path",
        type=Path,
        default=Path("src/explainer/compliance_rules.txt"),
        help="Text file path for external compliance rules (금융소비자보호법 문항 등)",
    )
    p.add_argument(
        "--enable-understanding-eval",
        action="store_true",
        help="Enable user-simulator/evaluator based understanding assessment",
    )
    p.add_argument(
        "--max-understanding-samples",
        type=int,
        default=200,
        help="Maximum recommendation items to run understanding evaluation on",
    )
    p.add_argument("--use-llm-user-simulator", action="store_true")
    p.add_argument("--use-llm-evaluator", action="store_true")
    p.add_argument("--simulator-model", type=str, default="gpt-5-mini")
    p.add_argument("--evaluator-model", type=str, default="gpt-5-mini")
    p.add_argument(
        "--simulator-prompt-path",
        type=Path,
        default=Path("src/explainer/simulator.txt"),
        help="Path to user simulator system prompt",
    )
    p.add_argument(
        "--evaluator-prompt-path",
        type=Path,
        default=Path("src/explainer/evaluator.txt"),
        help="Path to evaluator system prompt",
    )
    p.add_argument(
        "--understanding-log-jsonl",
        type=Path,
        default=Path("reports/e2e/explainer_understanding_logs.jsonl"),
        help="JSONL path for per-item understanding evaluation logs",
    )
    return parse_args_with_config(p, section="evaluate_explainer")


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

    eval_snapshots = sample_users(
        snapshots,
        user_col=cfg.user_key_11,
        max_users=args.max_eval_users,
        random_state=cfg.random_state,
    )

    llm_renderer = (
        OpenAILLMRenderer(model=args.llm_model, prompt_path=args.llm_prompt_path)
        if args.use_llm_renderer
        else None
    )
    explainer = GroundedExplainer(
        rec,
        llm_renderer=llm_renderer,
        fallback_to_template_on_verify_fail=not args.no_template_fallback,
        use_explainer_moe=bool(args.use_explainer_moe),
        compliance_rules_path=args.compliance_rules_path,
        explainer_moe_debug=bool(args.explainer_moe_debug),
    )

    understanding_evaluator = None
    if args.enable_understanding_eval:
        understanding_evaluator = ExplainerUnderstandingEvaluator(
            use_llm_user_simulator=bool(args.use_llm_user_simulator),
            use_llm_evaluator=bool(args.use_llm_evaluator),
            simulator_model=str(args.simulator_model),
            evaluator_model=str(args.evaluator_model),
            simulator_prompt_path=args.simulator_prompt_path,
            evaluator_prompt_path=args.evaluator_prompt_path,
        )

    batch = evaluate_explainer_batch(
        explainer=explainer,
        eval_snapshots=eval_snapshots,
        top_k=args.top_k,
        understanding_evaluator=understanding_evaluator,
        max_understanding_samples=int(args.max_understanding_samples),
    )

    understanding_eval = batch.get("understanding_eval", {})
    if args.enable_understanding_eval:
        records = understanding_eval.get("records", [])
        if isinstance(records, list) and records:
            export_jsonl(records, args.understanding_log_jsonl)

    report = {
        "config": {
            "fit": bool(args.fit),
            "family": str(args.family),
            "sample_users": int(args.sample_users),
            "max_eval_users": int(args.max_eval_users),
            "top_k": int(args.top_k),
            "use_moe_harness": bool(args.use_moe_harness),
            "moe_debug": bool(args.moe_debug),
            "moe_ranker_weight": float(args.moe_ranker_weight),
            "moe_baseline_weight": float(args.moe_baseline_weight),
            "moe_utility_weight": float(args.moe_utility_weight),
            "moe_deposit_baseline_boost": float(args.moe_deposit_baseline_boost),
            "moe_fund_utility_boost": float(args.moe_fund_utility_boost),
            "moe_low_risk_fund_penalty": float(args.moe_low_risk_fund_penalty),
            "use_llm_renderer": bool(args.use_llm_renderer),
            "llm_model": str(args.llm_model),
            "llm_prompt_path": str(args.llm_prompt_path),
            "template_fallback": bool(not args.no_template_fallback),
            "use_explainer_moe": bool(args.use_explainer_moe),
            "explainer_moe_debug": bool(args.explainer_moe_debug),
            "compliance_rules_path": str(args.compliance_rules_path),
            "enable_understanding_eval": bool(args.enable_understanding_eval),
            "max_understanding_samples": int(args.max_understanding_samples),
            "use_llm_user_simulator": bool(args.use_llm_user_simulator),
            "use_llm_evaluator": bool(args.use_llm_evaluator),
            "simulator_model": str(args.simulator_model),
            "evaluator_model": str(args.evaluator_model),
            "simulator_prompt_path": str(args.simulator_prompt_path),
            "evaluator_prompt_path": str(args.evaluator_prompt_path),
            "understanding_log_jsonl": str(args.understanding_log_jsonl),
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
        "understanding_eval": {
            "enabled": bool(understanding_eval.get("enabled", False)),
            "evaluated_count": int(understanding_eval.get("evaluated_count", 0)),
            "metrics": understanding_eval.get("metrics", {}),
            "sample_records": understanding_eval.get("sample_records", []),
            "log_path": (
                str(args.understanding_log_jsonl)
                if args.enable_understanding_eval and int(understanding_eval.get("evaluated_count", 0)) > 0
                else None
            ),
        },
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
    ug = float(report["understanding_eval"]["metrics"].get("understanding_gain", 0.0) or 0.0)
    if report["understanding_eval"]["enabled"] and ug <= 0:
        report["warnings"].append(
            "Understanding gain is non-positive; explanation may sound fluent but not improve user comprehension."
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
