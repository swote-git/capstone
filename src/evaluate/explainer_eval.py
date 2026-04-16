from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .explainer_understanding_eval import ExplainerUnderstandingEvaluator


def sample_users(df: pd.DataFrame, user_col: str, max_users: int, random_state: int) -> pd.DataFrame:
    users = df[user_col].drop_duplicates()
    if len(users) <= max_users:
        return df
    selected = users.sample(n=max_users, random_state=random_state)
    return df[df[user_col].isin(selected)].copy()


def _mean_or_zero(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def evaluate_explainer_batch(
    explainer: Any,
    eval_snapshots: pd.DataFrame,
    top_k: int,
    understanding_evaluator: Optional[ExplainerUnderstandingEvaluator] = None,
    max_understanding_samples: int = 0,
) -> Dict[str, object]:
    alignments: List[float] = []
    hallucinations: List[float] = []
    fact_consistency: List[float] = []
    pass_flags: List[float] = []
    forbidden_flags: List[float] = []
    reasons_per_item: List[int] = []
    reason_feature_counts: Dict[str, int] = {}
    reason_patterns: Dict[str, int] = {}
    failed_examples: List[Dict[str, object]] = []
    understanding_records: List[Dict[str, object]] = []
    understanding_limit = int(max_understanding_samples) if int(max_understanding_samples) > 0 else 0

    for _, user_row in eval_snapshots.iterrows():
        out = explainer.explain_top_k(user_row, k=top_k)
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

            if understanding_evaluator is not None:
                if understanding_limit <= 0 or len(understanding_records) < understanding_limit:
                    understanding_records.append(
                        understanding_evaluator.evaluate_recommendation(
                            user_id=str(out["user_id"]),
                            recommendation_item=rec_item,
                        )
                    )

    top_reason_features = sorted(reason_feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    understanding_summary: Dict[str, object] = {
        "enabled": bool(understanding_evaluator is not None),
        "evaluated_count": int(len(understanding_records)),
        "metrics": {},
        "sample_records": [],
        "records": understanding_records,
    }
    if understanding_evaluator is not None:
        understanding_summary["metrics"] = understanding_evaluator.summarize(understanding_records)
        compact_samples: List[Dict[str, object]] = []
        for r in understanding_records[:5]:
            compact_samples.append(
                {
                    "user_id": str(r.get("user_id", "")),
                    "product_id": str(r.get("product_id", "")),
                    "render_source": str(r.get("render_source", "")),
                    "total_before": int(r.get("total_before", 0)),
                    "total_after": int(r.get("total_after", 0)),
                    "scores_before": r.get("scores_before", {}),
                    "scores_after": r.get("scores_after", {}),
                    "understanding_gain": float(
                        (r.get("effect_scores", {}) or {}).get("understanding_gain", 0.0)
                    ),
                    "misinterpretation_rate": float(
                        (r.get("effect_scores", {}) or {}).get("misinterpretation_rate", 0.0)
                    ),
                }
            )
        understanding_summary["sample_records"] = compact_samples

    return {
        "coverage": {
            "evaluated_users": int(eval_snapshots.shape[0]),
            "evaluated_explanations": int(len(pass_flags)),
        },
        "metrics": {
            "reason_coverage_rc": _mean_or_zero(alignments),
            "hallucination_rate_hr": _mean_or_zero(hallucinations),
            "fact_consistency_rate": _mean_or_zero(fact_consistency),
            "verification_pass_rate": _mean_or_zero(pass_flags),
            "forbidden_claim_rate": _mean_or_zero(forbidden_flags),
            "avg_reasons_per_item": _mean_or_zero([float(x) for x in reasons_per_item]),
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
        "understanding_eval": understanding_summary,
    }
