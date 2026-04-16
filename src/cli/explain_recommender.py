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
    return parse_args_with_config(p, section="explain_recommender")


def main() -> None:
    args = parse_args()
    cfg = RecommenderConfig(data_root=args.data_root, top_k=args.top_k, recommender_family=args.family)
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
    )
    print(to_json(result))


if __name__ == "__main__":
    main()
