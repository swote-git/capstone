#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _run_module(module: str, forwarded_args: list[str]) -> None:
    sys.argv = [module, *forwarded_args]
    runpy.run_module(module, run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified entrypoint for thin-file recommendation pipelines")
    parser.add_argument(
        "pipeline",
        choices=[
            "recommend",
            "explain",
            "evaluate",
            "evaluate-explainer",
            "benchmark-llm",
            "improve",
            "tps-main",
            "evaluate-custom",
            "demo-new-user",
        ],
        help="Pipeline to run",
    )
    args, rest = parser.parse_known_args()

    mapping = {
        "recommend": "cli.run_recommender",
        "explain": "cli.explain_recommender",
        "evaluate": "cli.evaluate",
        "evaluate-explainer": "cli.evaluate_explainer",
        "benchmark-llm": "cli.benchmark_llm_models",
        "improve": "cli.improve_recommender_with_utility",
        "tps-main": "cli.TPS_Main_v2",
        "evaluate-custom": "cli.evaluate_custom_v2",
        "demo-new-user": "cli.demo_new_user_v2",
    }
    _run_module(mapping[args.pipeline], rest)


if __name__ == "__main__":
    # Keep root entrypoint stable even when called from outside repo root.
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    # Allow `python main.py <pipeline> --help` to show pipeline-specific help.
    if len(sys.argv) >= 3 and sys.argv[2] in {"-h", "--help"}:
        mapping = {
            "recommend": "cli.run_recommender",
            "explain": "cli.explain_recommender",
            "evaluate": "cli.evaluate",
            "evaluate-explainer": "cli.evaluate_explainer",
            "benchmark-llm": "cli.benchmark_llm_models",
            "improve": "cli.improve_recommender_with_utility",
            "tps-main": "cli.TPS_Main_v2",
            "evaluate-custom": "cli.evaluate_custom_v2",
            "demo-new-user": "cli.demo_new_user_v2",
        }
        pipeline = sys.argv[1]
        if pipeline in mapping:
            _run_module(mapping[pipeline], ["--help"])
            raise SystemExit(0)
    main()
