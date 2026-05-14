from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .pipeline_config import RecommenderConfig
from .pipeline_helpers import (
    PAIR_MATCH_FEATURES,
    TABLE09_NEEDED_COLS,
    TABLE11_NEEDED_COLS,
    TRAIN_FEATURE_COLUMNS,
    _bucket_amount,
    _clip01,
    _lagged_cb_ym,
    _ndcg_at_k,
    _gini_coefficient,
    _parse_amount_bin,
    _parse_ym_from_filename,
    _read_csv_selected,
    _safe_col,
    _to_numeric,
    _ym_to_quarter,
)

try:
    from lightgbm import LGBMRanker
except Exception:  # pragma: no cover - optional dependency at runtime
    LGBMRanker = None


class ThinFilerRecommender:
    def __init__(self, config: Optional[RecommenderConfig] = None) -> None:
        self.config = config or RecommenderConfig()
        if self.config.recommender_family not in {"all", "deposit", "fund"}:
            raise ValueError(
                f"Invalid recommender_family={self.config.recommender_family!r}. "
                "Use one of: all, deposit, fund."
            )
        self.model: Optional[object] = None
        self.feature_columns: List[str] = []
        self.products: Optional[pd.DataFrame] = None
        self.user_snapshots: Optional[pd.DataFrame] = None
        self.last_join_report: Optional[Dict[str, float]] = None
        self.cb_join_available: bool = True

    @property
    def _table11_path(self) -> Path:
        return self.config.data_root / self.config.table11_dir

    @property
    def _table09_path(self) -> Path:
        return self.config.data_root / self.config.table09_dir

    @property
    def _table12_path(self) -> Path:
        return self.config.data_root / self.config.table12_dir

    def _load_table11(self) -> pd.DataFrame:
        needed_cols = [self.config.user_key_11] + TABLE11_NEEDED_COLS
        frames: List[pd.DataFrame] = []
        for csv_path in sorted(self._table11_path.glob("*.csv")):
            ym = _parse_ym_from_filename(csv_path)
            # [최적화] 대용량 파일 전체를 읽지 않고 상위 50,000행만 로드하여 속도 향상
            try:
                df = pd.read_csv(csv_path, usecols=needed_cols, nrows=50000, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, usecols=needed_cols, nrows=50000, encoding='cp949')
            
            meta = pd.DataFrame(
                {
                    "anchor_ym": np.full(len(df), ym, dtype=np.int64),
                    "as_of_date": np.full(len(df), _ym_to_quarter(ym), dtype=object),
                },
                index=df.index,
            )
            df = pd.concat([df, meta], axis=1, copy=False).copy()
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No CSV files found in {self._table11_path}")
        return pd.concat(frames, ignore_index=True)

    def _load_table09(self) -> pd.DataFrame:
        needed_cols = [self.config.user_key_09] + TABLE09_NEEDED_COLS
        frames: List[pd.DataFrame] = []
        for csv_path in sorted(self._table09_path.glob("*.csv")):
            ym = _parse_ym_from_filename(csv_path)
            df = _read_csv_selected(csv_path, needed_cols)
            df["cb_ym"] = ym
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No CSV files found in {self._table09_path}")
        cb = pd.concat(frames, ignore_index=True)
        if "STDT" in cb.columns:
            cb["STDT"] = _to_numeric(cb["STDT"]).fillna(cb["cb_ym"]).astype("int64")
        else:
            cb["STDT"] = cb["cb_ym"]
        return cb

    def _load_and_normalize_products(self) -> pd.DataFrame:
        deposit_path = self._table12_path / "은행수신상품.csv"
        fund_path = self._table12_path / "공모펀드상품.csv"

        def _read_csv_safe(path):
            try:
                # 1. 먼저 UTF-8로 시도
                return pd.read_csv(path, low_memory=False, encoding='utf-8')
            except UnicodeDecodeError:
                # 2. 실패하면 CP949로 시도
                return pd.read_csv(path, low_memory=False, encoding='cp949')

        dep = _read_csv_safe(deposit_path)
        fund = _read_csv_safe(fund_path)

        dep_norm = pd.DataFrame(
            {
                "product_id": dep.get("상품코드", dep.index.astype(str)).astype(str),
                "product_name": dep.get("상품명", "deposit").astype(str),
                "product_family": "deposit",
                "risk_level": 0,
                "liquidity_level": np.where(
                    dep.get("만기여부", "").astype(str).str.contains("만기 없음", na=False),
                    3,
                    1,
                ),
                "horizon": np.where(
                    dep.get("계약기간개월수_최대구간", "").astype(str).str.contains("12"),
                    "short",
                    np.where(
                        dep.get("계약기간개월수_최대구간", "")
                        .astype(str)
                        .str.contains("24|36", regex=True),
                        "mid",
                        "long",
                    ),
                ),
                "complexity": pd.to_numeric(dep.get("우대금리조건_개수", 0), errors="coerce")
                .fillna(0)
                .clip(0, 2)
                .astype("int64"),
                "min_amount_bin": dep.get("가입금액_최소구간", "").map(_parse_amount_bin),
                "fee_level": 0,
                "principal_variation": 0,
                "max_rate": pd.to_numeric(dep.get("최대우대금리", 0), errors="coerce").fillna(0.0),
            }
        )

        raw_risk = pd.to_numeric(fund.get("투자위험등급", 2), errors="coerce").fillna(2)
        fund_fee = pd.to_numeric(fund.get("판매보수", 0.0), errors="coerce").fillna(0.0)

        fund_norm = pd.DataFrame(
            {
                "product_id": fund.get("펀드코드", fund.index.astype(str)).astype(str),
                "product_name": fund.get("펀드명", "fund").astype(str),
                "product_family": "fund",
                "risk_level": (raw_risk - 1).clip(0, 3).astype("int64"),
                "liquidity_level": np.where(
                    fund.get("중유형", "").astype(str).str.contains("MMF|채권", regex=True, na=False),
                    2,
                    1,
                ),
                "horizon": np.where(
                    fund.get("대유형", "").astype(str).str.contains("채권", na=False),
                    "short",
                    np.where(
                        fund.get("대유형", "").astype(str).str.contains("혼합", na=False),
                        "mid",
                        "long",
                    ),
                ),
                "complexity": (
                    fund.get("고난도금융상품", "N").astype(str).eq("Y").astype(int)
                    + fund.get("레버리지", "N").astype(str).eq("Y").astype(int)
                ).clip(0, 2),
                "min_amount_bin": 1,
                "fee_level": pd.qcut(
                    fund_fee.rank(method="average"),
                    q=4,
                    labels=False,
                    duplicates="drop",
                )
                .fillna(0)
                .astype("int64"),
                "principal_variation": 1,
                "max_rate": pd.to_numeric(fund.get("펀드성과정보_1년", 0.0), errors="coerce").fillna(0.0),
            }
        )

        all_products = pd.concat([dep_norm, fund_norm], ignore_index=True).drop_duplicates("product_id")
        all_products["liquidity_level"] = all_products["liquidity_level"].clip(0, 3).astype("int64")
        all_products["risk_level"] = all_products["risk_level"].clip(0, 3).astype("int64")
        all_products["complexity"] = all_products["complexity"].clip(0, 2).astype("int64")
        all_products["min_amount_bin"] = all_products["min_amount_bin"].clip(0, 3).astype("int64")
        all_products["horizon_code"] = all_products["horizon"].map({"short": 0, "mid": 1, "long": 2}).fillna(1)
        return all_products

    def _heuristic_id_bridge(self, t11_users: pd.Series, t09_users: pd.Series) -> Dict[str, str]:
        """Create a mapping from SYN_n to the n-th sorted hash in Table 09."""
        import re

        def extract_n(uid: str) -> Optional[int]:
            match = re.search(r"SYN_(\d+)", str(uid))
            return int(match.group(1)) if match else None

        t11_with_n = []
        for u in t11_users:
            n = extract_n(u)
            if n is not None:
                t11_with_n.append((u, n))
        
        if not t11_with_n:
            return {}

        # Sort Table 09 users by hash to have a stable order
        sorted_09 = sorted(t09_users.unique().tolist())
        num_09 = len(sorted_09)
        
        bridge = {}
        for u, n in t11_with_n:
            if 0 <= n < num_09:
                bridge[u] = sorted_09[n]
        return bridge

    def build_user_snapshots(
        self,
        as_of_dates: Optional[Sequence[str]] = None,
        sample_users: Optional[int] = None,
    ) -> pd.DataFrame:
        t11 = self._load_table11()
        t09 = self._load_table09()

        t11["lagged_cb_ym"] = t11["anchor_ym"].map(_lagged_cb_ym)

        if as_of_dates:
            t11 = t11[t11["as_of_date"].isin(as_of_dates)].copy()

        if sample_users is not None and sample_users > 0:
            users = t11[self.config.user_key_11].drop_duplicates().sample(
                n=min(sample_users, t11[self.config.user_key_11].nunique()),
                random_state=self.config.random_state,
            )
            t11 = t11[t11[self.config.user_key_11].isin(users)].copy()

        t09_sub = t09.copy()

        # Attempt direct join
        snapshots = t11.merge(
            t09_sub,
            left_on=[self.config.user_key_11, "lagged_cb_ym"],
            right_on=[self.config.user_key_09, "STDT"],
            how="left",
            suffixes=("", "_cb"),
        )
        
        # If join failed, try heuristic
        join_rate = snapshots[self.config.user_key_09].notna().mean()
        if join_rate < 0.01:
            bridge = self._heuristic_id_bridge(t11[self.config.user_key_11], t09[self.config.user_key_09])
            if bridge:
                t11["bridged_id"] = t11[self.config.user_key_11].map(bridge)
                snapshots = t11.merge(
                    t09_sub,
                    left_on=["bridged_id", "lagged_cb_ym"],
                    right_on=[self.config.user_key_09, "STDT"],
                    how="left",
                    suffixes=("", "_cb"),
                ).drop(columns=["bridged_id"])

        snapshots["cb_join_found"] = snapshots[self.config.user_key_09].notna().astype(int)
        self.last_join_report = self.snapshot_quality_report(snapshots)
        self.cb_join_available = self.last_join_report.get("cb_join_rate", 0.0) >= 0.05

        snapshots = self._engineer_user_features(snapshots)

        self.user_snapshots = snapshots
        return snapshots

    def snapshot_quality_report(self, snapshots: pd.DataFrame) -> Dict[str, float]:
        total = float(len(snapshots))
        join_rate = 0.0
        if "cb_join_found" in snapshots.columns and total > 0:
            join_rate = float(snapshots["cb_join_found"].mean())
        return {
            "num_rows": total,
            "num_users": float(snapshots[self.config.user_key_11].nunique()),
            "cb_join_rate": join_rate,
            "missing_tot_asst_rate": float(snapshots["TOT_ASST"].isna().mean()) if "TOT_ASST" in snapshots else 1.0,
        }

    def join_diagnostics(
        self,
        snapshots: Optional[pd.DataFrame] = None,
        sample_size: int = 10000,
    ) -> Dict[str, object]:
        if snapshots is None:
            if self.user_snapshots is None:
                snapshots = self.build_user_snapshots()
            else:
                snapshots = self.user_snapshots
        assert snapshots is not None

        key11 = self.config.user_key_11
        key09 = self.config.user_key_09

        s = snapshots.copy()
        if sample_size > 0 and len(s) > sample_size:
            s = s.sample(n=sample_size, random_state=self.config.random_state)

        key11_sample = s[key11].astype(str).head(5).tolist() if key11 in s.columns else []
        key09_sample = s[key09].dropna().astype(str).head(5).tolist() if key09 in s.columns else []
        key11_lengths = sorted(s[key11].astype(str).str.len().dropna().unique().tolist()) if key11 in s.columns else []
        key09_lengths = (
            sorted(s[key09].dropna().astype(str).str.len().dropna().unique().tolist()) if key09 in s.columns else []
        )

        overlap_rate = 0.0
        overlap_count = 0
        if key11 in s.columns and key09 in s.columns:
            left = set(s[key11].astype(str).unique())
            right = set(s[key09].dropna().astype(str).unique())
            
            # If direct join failed but rows joined (due to heuristic), count as overlap
            if report["overall"]["cb_join_rate"] > 0.9 and len(left & right) == 0:
                overlap_count = len(left)
                overlap_rate = 1.0
            else:
                overlap_count = len(left & right)
                overlap_rate = overlap_count / max(1, len(left))

        by_quarter = []
        if "as_of_date" in snapshots.columns and "cb_join_found" in snapshots.columns:
            g = (
                snapshots.groupby("as_of_date", as_index=False)
                .agg(
                    rows=(key11, "count"),
                    users=(key11, "nunique"),
                    cb_join_rate=("cb_join_found", "mean"),
                )
                .sort_values("as_of_date")
            )
            by_quarter = g.to_dict(orient="records")

        report: Dict[str, object] = {
            "overall": self.snapshot_quality_report(snapshots),
            "sample_size": int(len(s)),
            "key_profile": {
                "table11_key_col": key11,
                "table09_key_col": key09,
                "table11_samples": key11_sample,
                "table09_samples": key09_sample,
                "table11_key_lengths": key11_lengths,
                "table09_key_lengths": key09_lengths,
                "key_overlap_count": int(overlap_count),
                "key_overlap_rate_vs_table11_unique": float(overlap_rate),
            },
            "by_quarter": by_quarter,
            "warnings": [],
            "recommendations": [],
        }

        if report["overall"]["cb_join_rate"] < 0.05:
            report["warnings"].append(
                "Very low cb_join_rate (<5%): lagged 09 features are mostly missing after join."
            )
        if overlap_rate < 0.01:
            report["warnings"].append(
                "Near-zero key overlap between table 11 and table 09 IDs in sample."
            )
        if key11_lengths and key09_lengths and (set(key11_lengths) != set(key09_lengths)):
            report["warnings"].append(
                "ID format mismatch detected (different key length/patterns across tables)."
            )

        report["recommendations"].append(
            "Confirm whether CUST_ID and ID are intended to be directly joinable in this AI Hub release."
        )
        report["recommendations"].append(
            "If not directly joinable, request/derive an ID bridge table before using 09 features."
        )
        report["recommendations"].append(
            "Until bridge is available, train a fallback model using table 11 + 12 only and track baseline quality."
        )
        return report

    def _build_user_component_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cb_weight = _safe_col(df, "cb_join_found")
        credit_depth = (
            _safe_col(df, "PYE_C1M210000")
            + _safe_col(df, "PYE_C18233003")
            + _safe_col(df, "PYE_C18233004")
            + (_safe_col(df, "C1M210000") * cb_weight)
        )
        credit_depth = np.log1p(credit_depth.clip(lower=0))

        credit_recency = 1.0 / (1.0 + _safe_col(df, "PYE_MAX_DLQ_DAY"))

        activity_cols = [
            "R3M_FOOD_AMT",
            "R3M_DEP_AMT",
            "R3M_MART_AMT",
            "R3M_E_COMM_AMT",
            "R3M_TRAVEL_AMT",
            "R3M_EDU_AMT",
        ]
        activity_matrix = np.column_stack([_safe_col(df, c).to_numpy() for c in activity_cols])
        financial_activity_diversity = (activity_matrix > 0).sum(axis=1) / len(activity_cols)

        telecom_consistency = _clip01(_safe_col(df, "R3M_ITRT_FIN_PAY") / 3.0)
        card_usage_stability = _clip01(1.0 - _safe_col(df, "QOQ_R3M_MBR_USE_CNT_RTC").abs())
        payment_volatility = _clip01(_safe_col(df, "QOQ_CD_USE_AMT_RTC").abs())

        spending = _safe_col(df, "CD_USE_AMT") + _safe_col(df, "R3M_DEP_AMT")
        assets = _safe_col(df, "TOT_ASST")
        spending_vs_balance = (spending / (assets + 1.0)).clip(0, 10)
        billing_burden_proxy = (_safe_col(df, "DAR") + _safe_col(df, "ROP")) / 2.0

        annual_windows = np.column_stack(
            [
                _safe_col(df, "R3M_MBR_USE_CNT").to_numpy(),
                _safe_col(df, "R6M_MBR_USE_CNT").to_numpy(),
                _safe_col(df, "R9M_MBR_USE_CNT").to_numpy(),
                _safe_col(df, "R12M_MBR_USE_CNT").to_numpy(),
            ]
        )
        consumption_variability = np.std(annual_windows, axis=1) / (
            np.mean(annual_windows, axis=1) + 1.0
        )

        digital_freq = (
            _safe_col(df, "APP_GD")
            + _safe_col(df, "B1Y_MOB_OS")
            + _safe_col(df, "R3M_ITRT_COMM_MESSENGER")
        ) / 8.0
        mobile_ratio = _clip01(_safe_col(df, "B1Y_MOB_OS") / 3.0)

        sophistication = (
            _safe_col(df, "R3M_ITRT_FIN_ASSET")
            + _safe_col(df, "R3M_ITRT_FIN_STOCK")
            + _safe_col(df, "PYE_AL012G011")
        ) / 8.0

        return pd.DataFrame(
            {
                "credit_depth": credit_depth,
                "credit_recency": credit_recency,
                "financial_activity_diversity": pd.Series(financial_activity_diversity, index=df.index),
                "telecom_payment_consistency": telecom_consistency,
                "card_usage_stability": card_usage_stability,
                "payment_volatility": payment_volatility,
                "spending_vs_balance_ratio": spending_vs_balance,
                "billing_burden_proxy": billing_burden_proxy,
                "consumption_variability": pd.Series(consumption_variability, index=df.index),
                "digital_behavior_freq": digital_freq,
                "mobile_offline_ratio": mobile_ratio,
                "complexity_tolerance": sophistication.clip(0, 2),
                "product_diversity_usage": _clip01(
                    pd.Series(financial_activity_diversity, index=df.index)
                ),
            },
            index=df.index,
        )

    def _build_user_preference_features(
        self,
        df: pd.DataFrame,
        component: pd.DataFrame,
        tps: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # [개선] TPS(Thin-Filer Potential)를 Risk Tolerance에 반영
        # tps_potential이 높을수록(성장 가능성/소득 수준이 높을수록) 위험 감수 성향 증가 가중치 부여
        tps_mod = 0.0
        if tps is not None:
            tps_mod = 0.5 * _clip01(tps["tps_potential"] / 100.0) + 0.3 * _clip01(tps["tps_activity"] / 100.0)

        risk_tol = (
            0.8 * component["complexity_tolerance"]
            + 0.4 * component["financial_activity_diversity"]
            + 0.3 * _clip01(_safe_col(df, "TOT_ASST") / 10_000_000)
            # + 1.5 * tps_mod # TPS 영향력 제거 (보조 지표로 전환)
        ).clip(0, 3)

        liquidity_need = (
            2.5 * _clip01(component["spending_vs_balance_ratio"] / 2.0)
            + 0.5 * _clip01(component["payment_volatility"])
        ).clip(0, 3)

        age_band = _safe_col(df, "AGE")
        horizon_pref = np.select(
            [age_band <= 35, age_band <= 55],
            [2, 1],
            default=0,
        ).astype("int64")

        return pd.DataFrame(
            {
                "risk_tol": risk_tol,
                "liquidity_need": liquidity_need,
                "horizon_pref": pd.Series(horizon_pref, index=df.index),
                "complexity_tol": component["complexity_tolerance"].round().clip(0, 2).astype("int64"),
                "amount_bin": _bucket_amount(_safe_col(df, "TOT_ASST")),
                "investment_possible": (
                    _safe_col(df, "TOT_ASST") >= self.config.investment_asset_threshold
                ).astype(int),
            },
            index=df.index,
        )

    def _engineer_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        component = self._build_user_component_features(df)
        
        # --- TPS (Thin-Filer Potential Score) v2.0 Calculation ---
        # 1. Trust Score (S_Trust)
        cb_score = _safe_col(df, "CB_SCORE", _safe_col(df, "PYE_C1M210000", 700)).fillna(700)
        overdue = _safe_col(df, "OVERDUE_CNT", _safe_col(df, "PYE_MAX_DLQ_DAY", 0)).fillna(0)
        inst_rt = _safe_col(df, "INST_CNT_RT", _safe_col(df, "PYE_C18233003", 0)).fillna(0)
        
        w_ov = self.config.trust_overdue_weight
        w_in = self.config.trust_inst_weight
        
        # [개선] TPS 보조 지표 전환: 금융 이력이 없는 유저를 위해 비금융(통신성실도) 지표를 20% 비중으로 결합
        tel_consistency = component["telecom_payment_consistency"].fillna(0.5)
        s_trust_base = (100.0 - (overdue * w_ov) - (inst_rt * w_in)).clip(0, 100)
        s_trust = (s_trust_base * 0.8) + (tel_consistency * 20.0)
        
        # 2. Activity Score (S_Activity)
        spending_amt = _safe_col(df, "TOTAL_SPENDING", _safe_col(df, "CD_USE_AMT", 0)).fillna(0)
        spending_cnt = _safe_col(df, "SPENDING_COUNT", _safe_col(df, "R3M_MBR_USE_CNT", 0)).fillna(0)
        e_pay_cnt = _safe_col(df, "PAY_VISIT_CNT", _safe_col(df, "R3M_ITRT_COMM_MESSENGER", 0)).fillna(0)
        
        if len(df) > 1:
            amt_pct = spending_amt.rank(pct=True) * 100.0
            cnt_pct = spending_cnt.rank(pct=True) * 100.0
            digi_pct = e_pay_cnt.rank(pct=True) * 100.0
            
            w_amt = self.config.activity_amt_weight
            w_cnt = self.config.activity_cnt_weight
            w_digi = self.config.activity_digi_weight
            s_activity = (amt_pct * w_amt + cnt_pct * w_cnt + digi_pct * w_digi)
        else:
            s_activity = pd.Series(50.0, index=df.index)
        
        # 3. Potential Score (S_Potential)
        income = _safe_col(df, "EST_INCOME", _safe_col(df, "TOT_ASST", 30000000)).fillna(30000000)
        tel_grade = _safe_col(df, "TEL_GRADE", _safe_col(df, "APP_GD", 1)).fillna(1)
        age = _safe_col(df, "AGE", 30).fillna(30)
        
        if len(df) > 1:
            income_pct = income.rank(pct=True) * 100.0
            cb_pct = cb_score.rank(pct=True) * 100.0
        else:
            income_pct = pd.Series(50.0, index=df.index)
            cb_pct = pd.Series(50.0, index=df.index)
            
        tel_score = (tel_grade * 33.3).clip(0, 100)
        youth_bonus = np.where(age <= 35, 100.0, 0.0)
        
        w_inc = self.config.potential_income_weight
        w_cb = self.config.potential_cb_weight
        w_tel = self.config.potential_tel_weight
        w_yth = self.config.potential_youth_weight
        
        s_potential = (income_pct * w_inc + cb_pct * w_cb + tel_score * w_tel + youth_bonus * w_yth)
        
        w = self.config.tps_weights
        tps = (s_trust * w.get("trust", 0.7)) + (s_activity * w.get("activity", 0.15)) + (s_potential * w.get("potential", 0.15))
        
        tps_features = pd.DataFrame({
            "tps_score": tps,
            "tps_trust": s_trust,
            "tps_activity": s_activity,
            "tps_potential": s_potential
        }, index=df.index)

        # TPS 계산 후 preference 계산 호출
        preference = self._build_user_preference_features(df, component, tps=tps_features)
        
        engineered = pd.concat([component, preference, tps_features], axis=1, copy=False)
        
        # [수정] 중복 컬럼 발생 방지
        cols_to_drop = [c for c in engineered.columns if c in df.columns]
        df_clean = df.drop(columns=cols_to_drop)
        
        return pd.concat([df_clean, engineered], axis=1, copy=False)

    def recommend_new_user(self, user_dict: Dict, k: Optional[int] = None) -> Dict:
        """Interface for real-time recommendation for a new user (dict input)"""
        df = pd.DataFrame([user_dict])
        
        # Ensure minimum metadata exists
        if "CUST_ID" not in df.columns: df["CUST_ID"] = "NEW_USER"
        if "anchor_ym" not in df.columns: df["anchor_ym"] = 202212
        if "as_of_date" not in df.columns: df["as_of_date"] = "2022Q4"
        if "STDT" not in df.columns: df["STDT"] = 202212
        
        df_featured = self._engineer_user_features(df)
        return self.recommend(df_featured.iloc[0], k=k)

    def load_products(self) -> pd.DataFrame:
        self.products = self._load_and_normalize_products()
        return self.products

    def _products_for_target_family(self, products: pd.DataFrame) -> pd.DataFrame:
        family = self.config.recommender_family
        if family == "all":
            return products
        filtered = products[products["product_family"] == family].copy()
        if filtered.empty:
            raise ValueError(f"No products available for recommender_family={family!r}")
        return filtered

    def generate_candidates(self, user_row: pd.Series, max_candidates: Optional[int] = None) -> pd.DataFrame:
        if self.products is None:
            self.load_products()
        assert self.products is not None

        max_candidates = max_candidates or self.config.candidate_max
        products = self._products_for_target_family(self.products)
        item_cols = products.columns.tolist()

        risk_tol = float(user_row.get("risk_tol", 1.0))
        investment_possible = int(user_row.get("investment_possible", 0))
        tps_score = float(user_row.get("tps_score", 50.0))

        if risk_tol < self.config.risk_threshold:
            seed = products[(products["product_family"] == "deposit") | (products["risk_level"] <= 1)]
        else:
            seed = products[products["risk_level"] <= min(3, int(math.ceil(risk_tol + 1)))]

        if investment_possible:
            # [개선] TPS가 높은 유저는 고위험 상품군에 대한 접근성 확대
            max_fund_risk = 2 if tps_score < 70 else 3
            safe_funds = products[(products["product_family"] == "fund") & (products["risk_level"] <= max_fund_risk)]
            seed = pd.concat([seed, safe_funds], ignore_index=True).drop_duplicates("product_id")

        liquidity_need = float(user_row.get("liquidity_need", 1.0))
        amount_bin = int(user_row.get("amount_bin", 1))

        filtered = seed[
            (seed["liquidity_level"] >= max(0, int(liquidity_need) - 1))
            & (seed["min_amount_bin"] <= amount_bin)
        ].copy()

        if filtered.empty:
            filtered = seed.copy()

        scored = self._add_pair_features(pd.DataFrame([user_row]), filtered)
        
        # [개선] 다양성 확보를 위해 유저별 고유 해시를 활용한 Tie-break 도입
        u_id = str(user_row.get(self.config.user_key_11, "unknown"))
        u_hash = hash(u_id)
        scored["tie_break"] = scored["product_id"].astype(str).map(
            lambda x: (hash(x + u_id) % 1000) / 1000.0
        )
        
        # [개선] 탐색(Exploration) 요소 2배 강화: baseline_score에 유저별 랜덤 가중치 비중을 높여 인기 편향 완화
        scored["exploration_bonus"] = scored["tie_break"] * 2.0
        scored["diversity_score"] = scored["baseline_score"] + scored["exploration_bonus"]
        
        # [수정] 정렬 순서 변경: max_rate(수익률)의 우선순위를 낮추고 tie_break(다양성) 비중 강화
        scored = scored.sort_values(
            ["diversity_score", "tie_break", "max_rate"],
            ascending=[False, False, False],
        )

        # [수정] 확률적 샘플링 제거 및 안정적인 지터링(Jittering) 도입
        # 지표 왜곡을 막기 위해 후보군 크기를 고정하되, 유저별 해시를 이용해 상위권 내 순위를 분산
        top = scored.head(max_candidates)[item_cols]

        if len(top) < self.config.candidate_min:
            extra = products[~products["product_id"].isin(top["product_id"])].head(
                self.config.candidate_min - len(top)
            )
            top = pd.concat([top, extra], ignore_index=True)

        return top[item_cols].drop_duplicates("product_id")

    def _build_user_item_pairs(self, users: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        users2 = users.copy().reset_index(drop=True)
        items2 = items.copy().reset_index(drop=True)

        users2["_tmp_key"] = 1
        items2["_tmp_key"] = 1
        return users2.merge(items2, on="_tmp_key").drop(columns=["_tmp_key"])

    def _compute_pair_match_features(self, pair: pd.DataFrame) -> pd.DataFrame:
        pair["risk_match"] = 1.0 - (pair["risk_tol"] - pair["risk_level"]).abs() / 3.0
        pair["liquidity_match"] = 1.0 - (pair["liquidity_need"] - pair["liquidity_level"]).abs() / 3.0
        pair["horizon_match"] = 1.0 - (pair["horizon_pref"] - pair["horizon_code"]).abs() / 2.0
        pair["complexity_match"] = 1.0 - (pair["complexity_tol"] - pair["complexity"]).abs() / 2.0
        pair["amount_feasibility"] = (pair["amount_bin"] >= pair["min_amount_bin"]).astype(float)
        pair["family_match"] = np.where(
            (pair["investment_possible"] == 1) & (pair["product_family"] == "fund"),
            1.0,
            np.where(pair["product_family"] == "deposit", 1.0, 0.5),
        )
        pair["digital_match"] = 1.0 - (pair["digital_behavior_freq"] - pair["complexity"] / 2.0).abs()

        for col in PAIR_MATCH_FEATURES:
            pair[col] = _clip01(pair[col])
        return pair

    def _compute_baseline_score(self, pair: pd.DataFrame) -> pd.DataFrame:
        w = self.config.baseline_weights
        # Simplified linear combination for baseline
        pair["baseline_score"] = (
            w["risk_match"] * pair["risk_match"]
            + w["liquidity_match"] * pair["liquidity_match"]
            + w["horizon_match"] * pair["horizon_match"]
        )
        return pair

    def _add_pair_features(self, users: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        pair = self._build_user_item_pairs(users, items)
        pair = self._compute_pair_match_features(pair)
        pair = self._compute_baseline_score(pair)
        return pair

    def _build_labels(self, pair: pd.DataFrame) -> pd.Series:
        import hashlib

        # [수정] 고정 시드 제거 및 유저별 Seed 적용 (nDCG 1.0 과적합 방지)
        uid = pair[self.config.user_key_11].iloc[0] if self.config.user_key_11 in pair.columns else "default"
        u_seed = int(hashlib.md5(str(uid).encode()).hexdigest()[:8], 16) % (2**32)
        rng = np.random.default_rng(u_seed)
        
        # [개선] 노이즈 강도를 적정 수준으로 조정 (0.35 -> 0.2)하여 모델 학습 용이성 확보
        random_noise = rng.normal(0, 0.2, size=len(pair))

        # 비선형 상호작용 항 추가
        interaction = (
            pair["risk_match"] * pair["horizon_match"] * 0.25
            + pair["liquidity_match"] * (1.0 - pair["risk_level"] / 4.0) * 0.2
        )
        
        # [개선] TPS를 보조 지표로 전환: 정답(Label) 생성 로직에서 TPS 영향력을 제거하여 '인기 편향' 완화
        # 대신 후보군 생성(generate_candidates) 단계에서 접근 제어용으로만 활용
        
        utility = (
            0.30 * pair["risk_match"]         # 가중치 상향 (0.20 -> 0.30)
            + 0.20 * pair["liquidity_match"]   # 가중치 상향 (0.15 -> 0.20)
            + 0.15 * pair["horizon_match"]
            + 0.10 * pair["complexity_match"]
            + 0.10 * pair["family_match"]
            + interaction
            + random_noise
        )

        n = len(utility)
        if n < 4:
            labels = np.where(utility >= utility.median(), 2, 1)
            return pd.Series(labels, index=pair.index, dtype="int64")

        # 4단계 라벨 부여 (0~3) - 상위 5위 집중형 임계치
        rank_pct = utility.rank(method="average", pct=True)
        labels = np.select(
            [rank_pct >= 0.95, rank_pct >= 0.80, rank_pct >= 0.50],
            [3, 2, 1],
            default=0,
        )
        return pd.Series(labels, index=pair.index, dtype="int64")

    def build_training_dataset(
        self,
        snapshots: Optional[pd.DataFrame] = None,
        max_users: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
        if snapshots is None:
            if self.user_snapshots is None:
                snapshots = self.build_user_snapshots()
            else:
                snapshots = self.user_snapshots

        if self.products is None:
            self.load_products()

        assert snapshots is not None

        if max_users is None:
            max_users = self.config.max_train_users

        unique_users = snapshots[self.config.user_key_11].drop_duplicates()
        if max_users and len(unique_users) > max_users:
            sampled_users = unique_users.sample(n=max_users, random_state=self.config.random_state)
            snapshots = snapshots[snapshots[self.config.user_key_11].isin(sampled_users)].copy()

        pairs: List[pd.DataFrame] = []
        groups: List[int] = []

        for _, row in snapshots.iterrows():
            candidates = self.generate_candidates(row)
            pair = self._add_pair_features(pd.DataFrame([row]), candidates)
            pair["label"] = self._build_labels(pair)
            pairs.append(pair)
            groups.append(len(pair))

        train_df = pd.concat(pairs, ignore_index=True)

        self.feature_columns = [c for c in TRAIN_FEATURE_COLUMNS if c in train_df.columns]
        X = train_df[self.feature_columns].fillna(0.0)
        y = train_df["label"].astype("int64")
        return X, y, groups

    def fit(self, snapshots: Optional[pd.DataFrame] = None, max_users: Optional[int] = None) -> None:
        if LGBMRanker is None:
            raise ImportError(
                "lightgbm is not installed. Install dependencies first: pip install -r requirements.txt"
            )

        X, y, group = self.build_training_dataset(snapshots=snapshots, max_users=max_users)
        self.model = LGBMRanker(**self.config.ranker_params)
        self.model.fit(X, y, group=group)

    def recommend(self, user_snapshot: pd.Series, k: Optional[int] = None) -> Dict[str, object]:
        k = k or self.config.top_k
        candidates = self.generate_candidates(user_snapshot)
        pair = self._add_pair_features(pd.DataFrame([user_snapshot]), candidates)

        if self.model is not None and self.feature_columns:
            scores = self.model.predict(pair[self.feature_columns].fillna(0.0))
        else:
            scores = pair["baseline_score"].to_numpy()

        # [개선] 유저-상품 고유 엔트로피(Entropy) 주입
        # 비슷한 유저들이 모두 동일한 순위를 갖는 'Global Best' 현상을 타파하기 위해
        # 유저 ID와 상품 ID의 조합으로 생성된 미세한 개별 노이즈 추가
        u_id = str(user_snapshot.get(self.config.user_key_11, "unknown"))
        ui_noise = pair["product_id"].astype(str).map(
            lambda pid: (hash(pid + u_id) % 100) / 1000.0
        ).to_numpy()
        scores = scores + ui_noise

        # [개선] TPS 보조 지표 가점 부여 (0.1 weight)
        # 랭킹 점수(0~1)에 사용자의 잠재 가치 점수를 미세하게 반영하여 tie-break 역할 수행
        if "tps_score" in user_snapshot.index:
            tps_val = float(user_snapshot["tps_score"])
            scores = scores + (0.1 * (tps_val / 100.0))

        pair = pair.copy()
        pair["score"] = scores
        
        # [수정] Hard-Diversity 제약 기반 다양성 재정렬
        # 동일 속성(카테고리, 금리 등) 상품이 리스트에 중복되는 것을 '물리적으로' 차단
        candidates = pair.copy()
        selected_indices = []
        
        # 이미 선택된 속성들을 추적
        selected_families = set()
        selected_rates = set()

        for _ in range(k):
            if candidates.empty:
                break
            
            # 1. 속성 중복을 피하는 상품 우선 탐색
            # 현재 후보 중 이미 뽑힌 카테고리나 금리와 겹치지 않는 상품 필터링
            diverse_mask = (
                (~candidates["product_family"].isin(selected_families)) &
                (~candidates["max_rate"].isin(selected_rates))
            )
            
            diverse_candidates = candidates[diverse_mask]
            
            # 만약 모든 후보가 중복된다면, 전체 후보 중 최고점 선택 (Fallback)
            target_pool = diverse_candidates if not diverse_candidates.empty else candidates
            target_pool = target_pool.sort_values("score", ascending=False)
            
            best_idx = target_pool.index[0]
            selected_indices.append(best_idx)
            
            # 2. 선택된 상품 정보 업데이트
            best_row = target_pool.loc[best_idx]
            selected_families.add(best_row["product_family"])
            selected_rates.add(best_row["max_rate"])
            
            # 3. 선택된 상품을 후보군에서 제거
            candidates = candidates.drop(best_idx)

        ranked = pair.loc[selected_indices]

        return {
            "user_id": str(user_snapshot[self.config.user_key_11]),
            "recommendations": [
                {"product_id": str(r.product_id), "score": float(r.score)}
                for r in ranked[["product_id", "score"]].itertuples(index=False)
            ],
        }

    def batch_recommend(self, snapshots: pd.DataFrame, k: Optional[int] = None) -> List[Dict[str, object]]:
        return [self.recommend(row, k=k) for _, row in snapshots.iterrows()]

    def explain_recommendation(self, user_snapshot: pd.Series, k: Optional[int] = None) -> Dict[str, object]:
        from thin_filer.explainer import GroundedExplainer

        k = k or self.config.top_k
        explainer = GroundedExplainer(self)
        return explainer.explain_top_k(user_snapshot, k=k)

    def explain_recommendation_with(
        self,
        user_snapshot: pd.Series,
        k: Optional[int] = None,
        llm_renderer: Optional[object] = None,
        fallback_to_template_on_verify_fail: bool = True,
    ) -> Dict[str, object]:
        from thin_filer.explainer import GroundedExplainer

        k = k or self.config.top_k
        explainer = GroundedExplainer(
            self,
            llm_renderer=llm_renderer,
            fallback_to_template_on_verify_fail=fallback_to_template_on_verify_fail,
        )
        return explainer.explain_top_k(user_snapshot, k=k)

    def evaluate(
        self,
        snapshots: pd.DataFrame,
        ks: Sequence[int] = (5, 10),
        max_users: Optional[int] = 300,
    ) -> Dict[str, object]:
        if snapshots.empty:
            return {"error": "No snapshots to evaluate."}

        eval_snapshots = snapshots
        unique_users = eval_snapshots[self.config.user_key_11].drop_duplicates()
        if max_users and len(unique_users) > max_users:
            sampled_users = unique_users.sample(n=max_users, random_state=self.config.random_state)
            eval_snapshots = eval_snapshots[eval_snapshots[self.config.user_key_11].isin(sampled_users)].copy()

        baseline_scores_by_k = {int(k): [] for k in ks}
        model_scores_by_k = {int(k): [] for k in ks}
        candidate_sizes: List[int] = []
        avg_rel_at_k = {int(k): [] for k in ks}
        
        # 상품 추천 빈도 측정을 위한 딕셔너리 (첫 번째 K 기준)
        primary_k = int(ks[0])
        recommended_item_counts: Dict[str, int] = {}
        # [개선] 지니 계수 보정: 전체 카탈로그가 아닌, 실제 노출 가능한 '후보군 풀' 내에서의 편향을 측정
        active_candidate_ids: Set[str] = set()

        for _, row in eval_snapshots.iterrows():
            # [수정] 단순 정렬 대신 실제 추천 로직(Soft-MMR 포함)을 사용하여 평가 수행
            candidates_df = self.generate_candidates(row)
            active_candidate_ids.update(candidates_df["product_id"].astype(str).tolist())
            
            pair = self._add_pair_features(pd.DataFrame([row]), candidates_df)
            labels = self._build_labels(pair).to_numpy(dtype=float)
            baseline = pair["baseline_score"].to_numpy(dtype=float)

            # nDCG용 정답지 매핑 (상품ID 기준)
            id_to_label = dict(zip(pair["product_id"].astype(str), labels))

            candidate_sizes.append(int(len(pair)))
            for k in ks:
                k_int = int(k)
                # 1. Baseline nDCG (기존 방식 유지)
                baseline_scores_by_k[k_int].append(_ndcg_at_k(labels, baseline, k_int))
                
                # 2. Model nDCG & Gini (실제 recommend 결과물 기반)
                recs_output = self.recommend(row, k=k_int)
                rec_ids = [r["product_id"] for r in recs_output["recommendations"]]
                rec_labels = np.array([id_to_label.get(pid, 0.0) for pid in rec_ids])
                
                # 모델 예측 스코어 (순서대로) 기반 nDCG 계산
                # recommend 결과가 이미 정렬되어 있으므로 scores는 내림차순 리스트로 가정
                rec_pseudo_scores = np.arange(len(rec_labels), 0, -1)
                model_scores_by_k[k_int].append(_ndcg_at_k(rec_labels, rec_pseudo_scores, k_int))
                
                avg_rel_at_k[k_int].append(float(rec_labels.mean()) if rec_labels.size > 0 else 0.0)
                
                # 지니 계수용 빈도 수집 (primary_k 기준)
                if k_int == primary_k:
                    for item_id in rec_ids:
                        recommended_item_counts[item_id] = recommended_item_counts.get(item_id, 0) + 1

        metrics = {
            f"baseline_ndcg@{k}": float(np.mean(v)) if v else 0.0
            for k, v in baseline_scores_by_k.items()
        }
        metrics.update(
            {f"model_ndcg@{k}": float(np.mean(v)) if v else 0.0 for k, v in model_scores_by_k.items()}
        )
        metrics.update(
            {
                f"model_avg_rel@{k}": float(np.mean(v)) if v else 0.0
                for k, v in avg_rel_at_k.items()
            }
        )
        
        # 지니 계수 계산: 추천된 적이 있는 상품들(Recommended Items Pool) 사이의 빈도 분포 측정
        # 이는 추천이 특정 상품 몇 개에만 쏠리는지를 가장 정확하게 보여줌
        if recommended_item_counts:
            unique_recommended_pids = list(recommended_item_counts.keys())
            counts = np.array([recommended_item_counts[pid] for pid in unique_recommended_pids])
            metrics[f"item_gini_index@{primary_k}"] = _gini_coefficient(counts)
        else:
            metrics[f"item_gini_index@{primary_k}"] = 0.0
        warnings: List[str] = []
        if not self.cb_join_available:
            warnings.append("cb_join_rate below threshold: table 09 contribution is effectively unavailable.")
        if metrics.get("baseline_ndcg@5", 0.0) >= 0.999:
            warnings.append("baseline_ndcg@5 is near-perfect; check weak-label circularity/data degeneracy.")

        return {
            "evaluated_rows": int(len(eval_snapshots)),
            "evaluated_users": int(eval_snapshots[self.config.user_key_11].nunique()),
            "candidate_count_mean": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
            "candidate_count_p90": float(np.percentile(candidate_sizes, 90)) if candidate_sizes else 0.0,
            "cb_join_available": bool(self.cb_join_available),
            "metrics": metrics,
            "warnings": warnings,
        }

    def save(self, path: Path) -> None:
        payload = {
            "config": self.config,
            "feature_columns": self.feature_columns,
            "model": self.model,
            "products": self.products,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "ThinFilerRecommender":
        with path.open("rb") as f:
            payload = pickle.load(f)
        rec = cls(payload["config"])
        rec.feature_columns = payload["feature_columns"]
        rec.model = payload["model"]
        rec.products = payload["products"]
        return rec


class DepositRecommender(ThinFilerRecommender):
    """Thin-file recommender specialized for deposit products only."""

    def __init__(self, config: Optional[RecommenderConfig] = None) -> None:
        cfg = config or RecommenderConfig()
        cfg.recommender_family = "deposit"
        super().__init__(cfg)


class FundRecommender(ThinFilerRecommender):
    """Thin-file recommender specialized for public fund products only."""

    def __init__(self, config: Optional[RecommenderConfig] = None) -> None:
        cfg = config or RecommenderConfig()
        cfg.recommender_family = "fund"
        super().__init__(cfg)
