from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def sample_users(df: pd.DataFrame, user_col: str, max_users: int, random_state: int) -> pd.DataFrame:
    users = df[user_col].drop_duplicates()
    if len(users) <= max_users:
        return df
    selected = users.sample(n=max_users, random_state=random_state)
    return df[df[user_col].isin(selected)].copy()


def _mean_or_zero(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def evaluate_explainer_batch(explainer: Any, eval_snapshots: pd.DataFrame, top_k: int) -> Dict[str, object]:
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

    top_reason_features = sorted(reason_feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]

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
    }

