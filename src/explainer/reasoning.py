from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from common.helpers import TABLE09_NEEDED_COLS, TABLE11_NEEDED_COLS

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


MATCH_REASON_FEATURES = {
    "risk_match",
    "liquidity_match",
    "horizon_match",
    "complexity_match",
    "amount_feasibility",
    "family_match",
    "digital_match",
}

TPS_REASON_FEATURES = {"tps_score", "tps_trust", "tps_activity", "tps_potential"}


def _normalize_feature_value(feature: str, value: float) -> float:
    """Normalize heterogeneous feature scales to roughly 0~1 for fallback contribution logic."""
    v = float(value)
    if feature in MATCH_REASON_FEATURES:
        return float(np.clip(v, 0.0, 1.0))
    if feature in TPS_REASON_FEATURES:
        return float(np.clip(v / 100.0, 0.0, 1.0))
    if feature in {"risk_level", "liquidity_level"}:
        return float(np.clip(v / 3.0, 0.0, 1.0))
    if feature in {"complexity", "complexity_tol"}:
        return float(np.clip(v / 2.0, 0.0, 1.0))
    if feature in {"min_amount_bin", "amount_bin"}:
        return float(np.clip(v / 3.0, 0.0, 1.0))
    if feature in {"investment_possible", "principal_variation"}:
        return float(np.clip(v, 0.0, 1.0))
    if feature in {"horizon_pref", "horizon_code"}:
        return float(np.clip(v / 2.0, 0.0, 1.0))
    if feature == "max_rate":
        # Symmetric squash for wide-return ranges (e.g., fund performance)
        return float(0.5 + 0.5 * np.tanh(v / 10.0))
    return float(np.clip(v, 0.0, 1.0))


def _select_reason_features(
    ranked: Sequence[tuple[str, float]],
    top_reason_k: int,
) -> List[tuple[str, float]]:
    """Diversify reasons: prioritize match features and keep TPS as supporting signal."""
    selected: List[tuple[str, float]] = []
    used = set()

    def pick(predicate, limit: int) -> None:
        nonlocal selected
        if limit <= 0:
            return
        for feature, contrib in ranked:
            if len(selected) >= top_reason_k:
                return
            if feature in used:
                continue
            if predicate(feature):
                selected.append((feature, contrib))
                used.add(feature)
                limit -= 1
                if limit <= 0:
                    return

    # 1) Start with customer-product fit signals (main explanation axis)
    pick(lambda f: f in MATCH_REASON_FEATURES, min(2, top_reason_k))
    # 2) Allow at most one TPS reason (supporting, not dominant)
    if len(selected) < top_reason_k:
        pick(lambda f: f in TPS_REASON_FEATURES, 1)
    # 3) Fill remaining slots from other strong features
    if len(selected) < top_reason_k:
        pick(lambda f: True, top_reason_k - len(selected))

    return selected


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
            normalized = np.asarray(
                [_normalize_feature_value(f, v) for f, v in zip(feature_cols, values)],
                dtype=float,
            )
            signed = normalized - 0.5
            return {f: float(g * s) for f, g, s in zip(feature_cols, gains, signed)}

    normalized = np.asarray(
        [_normalize_feature_value(f, v) for f, v in zip(feature_cols, values)],
        dtype=float,
    )
    return {f: float(v - 0.5) for f, v in zip(feature_cols, normalized)}


def extract_reasons(rec: Any, pair_row: pd.Series, top_reason_k: int) -> List[ReasonSignal]:
    feature_cols = top_feature_cols(rec)
    values = np.array([float(pair_row.get(c, 0.0)) for c in feature_cols], dtype=float)
    contributions = local_contributions(rec, feature_cols, values)

    ranked = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    selected = _select_reason_features(ranked, top_reason_k)
    signals: List[ReasonSignal] = []
    for feature, contrib in selected:
        value = float(pair_row.get(feature, 0.0))
        impact = "positive" if contrib >= 0 else "negative"
        signals.append(ReasonSignal(feature=feature, value=value, impact=impact, contribution=float(contrib)))
    return signals


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _extract_user_source_data(user_snapshot: pd.Series) -> Dict[str, Any]:
    table11_cols = [c for c in TABLE11_NEEDED_COLS if c in user_snapshot.index]
    table09_cols = [c for c in TABLE09_NEEDED_COLS if c in user_snapshot.index]

    table11_raw = {col: _to_jsonable(user_snapshot.get(col)) for col in table11_cols}
    table09_raw = {col: _to_jsonable(user_snapshot.get(col)) for col in table09_cols}

    join_context = {
        "CUST_ID": _to_jsonable(user_snapshot.get("CUST_ID")),
        "ID": _to_jsonable(user_snapshot.get("ID")),
        "as_of_date": _to_jsonable(user_snapshot.get("as_of_date")),
        "anchor_ym": _to_jsonable(user_snapshot.get("anchor_ym")),
        "lagged_cb_ym": _to_jsonable(user_snapshot.get("lagged_cb_ym")),
        "cb_join_found": _to_jsonable(user_snapshot.get("cb_join_found")),
    }

    return {
        "table11_raw": table11_raw,
        "table09_raw": table09_raw,
        "join_context": join_context,
    }


def retrieve_product_facts(rec: Any, pair_row: pd.Series) -> Dict[str, Any]:
    risk_level = float(pair_row.get("risk_level", 1))
    liquidity_level = float(pair_row.get("liquidity_level", 1))
    complexity = float(pair_row.get("complexity", 1))
    horizon_code = float(pair_row.get("horizon_code", 1))
    product_id = str(pair_row.get("product_id", ""))
    source_lookup = getattr(rec, "product_source_lookup", {}) or {}
    source_product_data = source_lookup.get(product_id, {})

    return {
        "product_id": product_id,
        "product_name": str(pair_row.get("product_name", "")),
        "family": str(pair_row.get("product_family", "unknown")),
        "risk": risk_label(risk_level),
        "liquidity": liquidity_label(liquidity_level),
        "horizon": horizon_label(horizon_code),
        "complexity": complexity_label(complexity),
        "principal_variation": bool(int(float(pair_row.get("principal_variation", 0)))),
        "product_meta": {
            "risk_level": risk_level,
            "liquidity_level": liquidity_level,
            "horizon_code": horizon_code,
            "complexity": complexity,
            "min_amount_bin": float(pair_row.get("min_amount_bin", 0)),
            "fee_level": float(pair_row.get("fee_level", 0)),
            "max_rate": float(pair_row.get("max_rate", 0)),
        },
        "match_detail": {
            "risk_match": float(pair_row.get("risk_match", 0)),
            "liquidity_match": float(pair_row.get("liquidity_match", 0)),
            "horizon_match": float(pair_row.get("horizon_match", 0)),
            "complexity_match": float(pair_row.get("complexity_match", 0)),
            "amount_feasibility": float(pair_row.get("amount_feasibility", 0)),
            "family_match": float(pair_row.get("family_match", 0)),
            "digital_match": float(pair_row.get("digital_match", 0)),
        },
        "source_product_data": source_product_data,
    }


def build_explanation_object(
    user_snapshot: pd.Series,
    product_facts: Dict[str, Any],
    reason_signals: Sequence[ReasonSignal],
    ranking_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    complexity_raw = float(
        user_snapshot.get(
            "complexity_tolerance",
            user_snapshot.get("complexity_tol", 1.0),
        )
    )
    user_summary = {
        "risk_preference": risk_label(float(user_snapshot.get("risk_tol", 1.0))),
        "liquidity_need": liquidity_label(float(user_snapshot.get("liquidity_need", 1.0))),
        "financial_knowledge": complexity_label(complexity_raw),
    }

    model_reasons = [reason_sentence(sig, user_summary, product_facts) for sig in reason_signals]

    alt_family = "fund" if product_facts["family"] == "deposit" else "deposit"
    if product_facts["family"] == "deposit":
        difference = "수익 잠재력은 높지만 위험과 원금 변동 가능성이 커질 수 있습니다"
    else:
        difference = "원금 안정성은 높지만 일반적으로 수익 잠재력은 낮아질 수 있습니다"

    warnings = warnings_from_facts(product_facts)
    if (
        product_facts.get("family") == "fund"
        and user_summary["risk_preference"] in {"낮음", "보통"}
        and product_facts.get("risk") in {"높음", "매우 높음"}
    ):
        warnings.insert(0, "고객님 성향 대비 위험이 높은 상품이므로 손실 가능성을 먼저 확인해 주세요.")
    caution_first = (
        product_facts.get("family") == "fund"
        and user_summary["risk_preference"] in {"낮음", "보통"}
        and product_facts.get("risk") in {"높음", "매우 높음"}
    )
    user_source_data = _extract_user_source_data(user_snapshot)
    product_source_data = product_facts.get("source_product_data", {})

    return {
        "user_summary": user_summary,
        "user_profile_detail": {
            "as_of_date": str(user_snapshot.get("as_of_date", "")),
            "risk_tol_score": float(user_snapshot.get("risk_tol", 0)),
            "liquidity_need_score": float(user_snapshot.get("liquidity_need", 0)),
            "horizon_pref_code": int(float(user_snapshot.get("horizon_pref", 1))),
            "complexity_tol_score": complexity_raw,
            "amount_bin": int(float(user_snapshot.get("amount_bin", 0))),
            "investment_possible": bool(int(float(user_snapshot.get("investment_possible", 0)))),
            "digital_behavior_freq": float(user_snapshot.get("digital_behavior_freq", 0)),
            "credit_depth": float(user_snapshot.get("credit_depth", 0)),
            "telecom_payment_consistency": float(user_snapshot.get("telecom_payment_consistency", 0)),
            "card_usage_stability": float(user_snapshot.get("card_usage_stability", 0)),
            "spending_vs_balance_ratio": float(user_snapshot.get("spending_vs_balance_ratio", 0)),
            "tps_score": float(user_snapshot.get("tps_score", 0)),
            "tps_trust": float(user_snapshot.get("tps_trust", 0)),
            "tps_activity": float(user_snapshot.get("tps_activity", 0)),
            "tps_potential": float(user_snapshot.get("tps_potential", 0)),
        },
        "recommended_product": {
            "product_id": product_facts.get("product_id", ""),
            "product_name": product_facts.get("product_name", ""),
            "family": product_facts["family"],
            "risk": product_facts["risk"],
            "liquidity": product_facts["liquidity"],
        },
        "recommended_product_detail": {
            "horizon": product_facts.get("horizon", ""),
            "complexity": product_facts.get("complexity", ""),
            "principal_variation": bool(product_facts.get("principal_variation", False)),
            "product_meta": product_facts.get("product_meta", {}),
            "match_detail": product_facts.get("match_detail", {}),
            "product_source_data": product_source_data,
        },
        "user_source_data": user_source_data,
        "product_source_data": product_source_data,
        "reason_signals": [
            {
                "feature": s.feature,
                "value": float(s.value),
                "impact": s.impact,
                "contribution": float(s.contribution),
            }
            for s in reason_signals
        ],
        "model_reasons": model_reasons,
        "comparison": {
            "alternative": alt_family,
            "difference": difference,
        },
        "warnings": warnings,
        "ranking_context": ranking_context or {},
        "explanation_policy": {
            "caution_first": bool(caution_first),
            "hide_internal_feature_names": True,
            "hide_raw_long_decimals": True,
        },
    }
