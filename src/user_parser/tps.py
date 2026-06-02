from __future__ import annotations

import re
from typing import Dict

import numpy as np
import pandas as pd


def compute_tps_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tel_score = out["TEL_GRADE"] * 33.3
    s_trust_base = (100.0 - (out["OVERDUE_CNT"] * 10.0) - (out["INST_CNT"] * 1.0)).clip(0, 100)
    out["s_trust"] = (s_trust_base * 0.8) + (tel_score.clip(0, 100) * 0.2)
    out["s_activity"] = (
        out["TOTAL_SPENDING"].rank(pct=True) * 20
        + out["SPENDING_COUNT"].rank(pct=True) * 40
        + out["PAY_VISIT_CNT"].rank(pct=True) * 40
    )
    income_pct = out["EST_INCOME"].rank(pct=True) * 100.0
    cb_pct = out["CB_SCORE"].rank(pct=True) * 100.0

    if "AGE_GB" in out.columns:
        youth_bonus = out["AGE_GB"].apply(lambda x: 100.0 if x in ["20대", "30대"] else 0.0)
    else:
        age = pd.to_numeric(out.get("AGE", 30), errors="coerce").fillna(30)
        youth_bonus = np.where(age <= 35, 100.0, 0.0)

    out["s_potential"] = income_pct * 0.3 + cb_pct * 0.4 + tel_score.clip(0, 100) * 0.1 + youth_bonus * 0.2
    out["tps_score"] = (out["s_trust"] * 0.7) + (out["s_activity"] * 0.15) + (out["s_potential"] * 0.15)
    return out


def _age_from_age_gb(age_gb: object, default: float = 30.0) -> float:
    if age_gb is None:
        return default
    s = str(age_gb).strip()
    if not s:
        return default
    m = re.search(r"(\d{2})\s*대", s)
    if m:
        return float(int(m.group(1)) + 5)
    return default


def parse_custom_user_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw_age_gb = out["AGE_GB"].copy() if "AGE_GB" in out.columns else pd.Series(np.nan, index=out.index)
    for col in out.columns:
        if col not in {"CUST_ID", "AGE_GB"}:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["user_id"] = out["CUST_ID"]
    if "AGE_GB" in out.columns:
        out["AGE_GB"] = raw_age_gb

    if "AGE" in out.columns:
        age_src = pd.to_numeric(out["AGE"], errors="coerce")
    else:
        age_src = pd.Series(np.nan, index=out.index)
    age_from_band = raw_age_gb.apply(lambda x: _age_from_age_gb(x, default=np.nan))
    out["AGE"] = age_src.fillna(age_from_band).fillna(30)

    raw_tps_cols = {
        "EST_INCOME",
        "CB_SCORE",
        "TOTAL_SPENDING",
        "SPENDING_COUNT",
        "OVERDUE_CNT",
        "INST_CNT",
        "PAY_VISIT_CNT",
        "TEL_GRADE",
    }
    if raw_tps_cols.issubset(out.columns):
        out = compute_tps_scores(out)

    s_trust = out["s_trust"] if "s_trust" in out.columns else pd.Series(50.0, index=out.index)
    s_activity = out["s_activity"] if "s_activity" in out.columns else pd.Series(50.0, index=out.index)
    s_potential = out["s_potential"] if "s_potential" in out.columns else pd.Series(50.0, index=out.index)
    tps_score = out["tps_score"] if "tps_score" in out.columns else (0.7 * s_trust + 0.15 * s_activity + 0.15 * s_potential)

    out["risk_tol"] = (out["CB_SCORE"] / 1000 * 1.5 + (s_potential / 100) * 1.5).clip(0, 3)
    out["liquidity_need"] = (2.0 - (s_trust / 100 * 1.5)).clip(0, 2)
    out["horizon_pref"] = 1.0
    out["complexity_tol"] = ((s_trust / 100) * 2.0).clip(0, 2)
    out["amount_bin"] = (out["EST_INCOME"].fillna(0) / 10000000).astype(int).clip(0, 3)
    out["investment_possible"] = 1.0
    out["digital_behavior_freq"] = (s_activity / 100).clip(0, 1)
    out["credit_depth"] = (out["CB_SCORE"] / 1000).clip(0, 1)
    out["credit_recency"] = 0.8
    out["telecom_payment_consistency"] = 0.9
    out["card_usage_stability"] = (s_trust / 100).clip(0, 1)
    out["spending_vs_balance_ratio"] = 0.5
    out["tps_score"] = pd.to_numeric(tps_score, errors="coerce").fillna(50.0)
    out["tps_trust"] = pd.to_numeric(s_trust, errors="coerce").fillna(50.0)
    out["tps_activity"] = pd.to_numeric(s_activity, errors="coerce").fillna(50.0)
    out["tps_potential"] = pd.to_numeric(s_potential, errors="coerce").fillna(50.0)
    out["C1M210000"] = out["CB_SCORE"]
    out["CD_USE_AMT"] = out["TOTAL_SPENDING"]
    out["TOT_ASST"] = out["EST_INCOME"]
    out["R3M_MBR_USE_CNT"] = out["SPENDING_COUNT"]
    out["B1Y_MOB_OS"] = out["TEL_GRADE"]
    out["anchor_ym"] = 202212
    out["as_of_date"] = "2022Q4"
    out["STDT"] = 202212
    return out


def parse_new_user(user_dict: Dict) -> Dict:
    out = dict(user_dict)
    out.setdefault("CUST_ID", "NEW_USER")
    out.setdefault("anchor_ym", 202212)
    out.setdefault("as_of_date", "2022Q4")
    out.setdefault("STDT", 202212)
    return out
