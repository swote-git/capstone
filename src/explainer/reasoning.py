from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .common import (
    ReasonSignal,
    complexity_label,
    horizon_label,
    liquidity_label,
    reason_sentence,
    risk_label,
    top_feature_cols,
    warnings_from_facts,
)

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


def local_contributions(rec: Any, feature_cols: Sequence[str], values: np.ndarray) -> Dict[str, float]:
    if rec.model is not None and shap is not None and len(feature_cols) > 0:
        try:
            explainer = shap.TreeExplainer(rec.model)
            x = pd.DataFrame([values], columns=list(feature_cols))
            shap_values = explainer.shap_values(x)
            if isinstance(shap_values, list):
                local = np.asarray(shap_values[0])[0]
            else:
                local = np.asarray(shap_values)[0]
            return {f: float(v) for f, v in zip(feature_cols, local)}
        except Exception:
            pass

    if rec.model is not None and hasattr(rec.model, "feature_importances_"):
        gains = np.asarray(rec.model.feature_importances_, dtype=float)
        if gains.size == len(feature_cols) and gains.sum() > 0:
            gains = gains / gains.sum()
            signed = values - 0.5
            return {f: float(g * s) for f, g, s in zip(feature_cols, gains, signed)}

    return {f: float(v - 0.5) for f, v in zip(feature_cols, values)}


def extract_reasons(rec: Any, pair_row: pd.Series, top_reason_k: int) -> List[ReasonSignal]:
    feature_cols = top_feature_cols(rec)
    values = np.array([float(pair_row.get(c, 0.0)) for c in feature_cols], dtype=float)
    contributions = local_contributions(rec, feature_cols, values)

    ranked = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    signals: List[ReasonSignal] = []
    for feature, contrib in ranked[:top_reason_k]:
        value = float(pair_row.get(feature, 0.0))
        impact = "positive" if contrib >= 0 else "negative"
        signals.append(ReasonSignal(feature=feature, value=value, impact=impact, contribution=float(contrib)))
    return signals


def retrieve_product_facts(pair_row: pd.Series) -> Dict[str, Any]:
    return {
        "family": str(pair_row.get("product_family", "unknown")),
        "risk": risk_label(float(pair_row.get("risk_level", 1))),
        "liquidity": liquidity_label(float(pair_row.get("liquidity_level", 1))),
        "horizon": horizon_label(float(pair_row.get("horizon_code", 1))),
        "complexity": complexity_label(float(pair_row.get("complexity", 1))),
        "principal_variation": bool(int(float(pair_row.get("principal_variation", 0)))),
    }


def build_explanation_object(
    user_snapshot: pd.Series,
    product_facts: Dict[str, Any],
    reason_signals: Sequence[ReasonSignal],
) -> Dict[str, Any]:
    user_summary = {
        "risk_preference": risk_label(float(user_snapshot.get("risk_tol", 1.0))),
        "liquidity_need": liquidity_label(float(user_snapshot.get("liquidity_need", 1.0))),
        "financial_knowledge": complexity_label(float(user_snapshot.get("complexity_tolerance", 1.0))),
    }

    model_reasons = [reason_sentence(sig, user_summary, product_facts) for sig in reason_signals]

    alt_family = "fund" if product_facts["family"] == "deposit" else "deposit"
    if product_facts["family"] == "deposit":
        difference = "higher return potential but higher risk and principal fluctuation"
    else:
        difference = "more capital stability but typically lower return potential"

    warnings = warnings_from_facts(product_facts)

    return {
        "user_summary": user_summary,
        "recommended_product": {
            "family": product_facts["family"],
            "risk": product_facts["risk"],
            "liquidity": product_facts["liquidity"],
        },
        "model_reasons": model_reasons,
        "comparison": {
            "alternative": alt_family,
            "difference": difference,
        },
        "warnings": warnings,
    }
