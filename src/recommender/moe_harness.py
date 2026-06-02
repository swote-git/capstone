from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def _safe_minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.zeros_like(arr, dtype=float)
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


def _normalized_weight_triplet(ranker: float, baseline: float, utility: float) -> Tuple[float, float, float]:
    vec = np.array([float(ranker), float(baseline), float(utility)], dtype=float)
    vec = np.clip(vec, 0.0, None)
    s = float(vec.sum())
    if s <= 0.0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    vec /= s
    return float(vec[0]), float(vec[1]), float(vec[2])


@dataclass
class MoEHarnessResult:
    final_scores: np.ndarray
    ranker_scores_norm: np.ndarray
    baseline_scores_norm: np.ndarray
    utility_scores_norm: np.ndarray
    weight_ranker: np.ndarray
    weight_baseline: np.ndarray
    weight_utility: np.ndarray

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "experts": ["ranker", "baseline", "utility"],
            "mean_weight_ranker": float(np.mean(self.weight_ranker)) if self.weight_ranker.size else 0.0,
            "mean_weight_baseline": float(np.mean(self.weight_baseline)) if self.weight_baseline.size else 0.0,
            "mean_weight_utility": float(np.mean(self.weight_utility)) if self.weight_utility.size else 0.0,
            "mean_score_ranker": float(np.mean(self.ranker_scores_norm)) if self.ranker_scores_norm.size else 0.0,
            "mean_score_baseline": float(np.mean(self.baseline_scores_norm)) if self.baseline_scores_norm.size else 0.0,
            "mean_score_utility": float(np.mean(self.utility_scores_norm)) if self.utility_scores_norm.size else 0.0,
        }


class MoEHarness:
    """Mixture-of-Experts harness for recommendation scoring.

    Experts:
    - `ranker`: trained LightGBM ranker score (if available)
    - `baseline`: deterministic baseline score
    - `utility`: lightweight utility proxy from pair features

    Router:
    - row-wise weights with family/risk-aware adjustments
    """

    def score_pair(self, rec: Any, user_row: pd.Series, pair: pd.DataFrame) -> MoEHarnessResult:
        n = len(pair)
        if n == 0:
            z = np.array([], dtype=float)
            return MoEHarnessResult(z, z, z, z, z, z, z)

        cfg = rec.config
        base_w = cfg.moe_default_weights or {}
        wr, wb, wu = _normalized_weight_triplet(
            base_w.get("ranker", 0.60),
            base_w.get("baseline", 0.25),
            base_w.get("utility", 0.15),
        )

        baseline_raw = pd.to_numeric(pair.get("baseline_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        baseline_norm = _safe_minmax(_clip01(baseline_raw))

        if rec.model is not None and rec.feature_columns:
            ranker_raw = rec.model.predict(pair[rec.feature_columns].fillna(0.0))
            ranker_norm = _safe_minmax(np.asarray(ranker_raw, dtype=float))
        else:
            ranker_norm = baseline_norm.copy()

        # Utility expert: pair-level rule score independent from label creation.
        utility_raw = (
            0.30 * pd.to_numeric(pair.get("risk_match", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            + 0.25 * pd.to_numeric(pair.get("liquidity_match", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            + 0.20 * pd.to_numeric(pair.get("horizon_match", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            + 0.15 * pd.to_numeric(pair.get("complexity_match", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            + 0.10 * pd.to_numeric(pair.get("amount_feasibility", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        if "hybrid_utility_score" in pair.columns:
            hus = pd.to_numeric(pair["hybrid_utility_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            utility_raw = 0.60 * utility_raw + 0.40 * hus
        utility_norm = _safe_minmax(_clip01(utility_raw))

        w_ranker = np.full(n, wr, dtype=float)
        w_baseline = np.full(n, wb, dtype=float)
        w_utility = np.full(n, wu, dtype=float)

        fam = pair.get("product_family", pd.Series([""] * n)).astype(str).to_numpy()
        dep_mask = fam == "deposit"
        fund_mask = fam == "fund"

        if dep_mask.any():
            w_baseline[dep_mask] += float(cfg.moe_deposit_baseline_boost)

        if fund_mask.any():
            w_utility[fund_mask] += float(cfg.moe_fund_utility_boost)

        risk_tol = float(pd.to_numeric(pd.Series([user_row.get("risk_tol", 1.5)]), errors="coerce").fillna(1.5).iloc[0])
        if risk_tol <= float(cfg.risk_threshold) and fund_mask.any():
            penalty = float(cfg.moe_low_risk_fund_penalty)
            w_ranker[fund_mask] = np.clip(w_ranker[fund_mask] - penalty * 0.5, 0.0, None)
            w_utility[fund_mask] = np.clip(w_utility[fund_mask] - penalty * 0.5, 0.0, None)
            w_baseline[fund_mask] = np.clip(w_baseline[fund_mask] + penalty, 0.0, None)

        # Row-wise normalization after routing adjustments.
        w_sum = w_ranker + w_baseline + w_utility
        w_sum = np.where(w_sum <= 0.0, 1.0, w_sum)
        w_ranker = w_ranker / w_sum
        w_baseline = w_baseline / w_sum
        w_utility = w_utility / w_sum

        final_scores = (
            w_ranker * ranker_norm
            + w_baseline * baseline_norm
            + w_utility * utility_norm
        )

        return MoEHarnessResult(
            final_scores=final_scores,
            ranker_scores_norm=ranker_norm,
            baseline_scores_norm=baseline_norm,
            utility_scores_norm=utility_norm,
            weight_ranker=w_ranker,
            weight_baseline=w_baseline,
            weight_utility=w_utility,
        )

