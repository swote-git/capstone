from __future__ import annotations

import math
import pickle
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from common.config import RecommenderConfig
from common.helpers import (
    PAIR_MATCH_FEATURES,
    TABLE09_NEEDED_COLS,
    TABLE11_NEEDED_COLS,
    TRAIN_FEATURE_COLUMNS,
    _bucket_amount,
    _clip01,
    _gini_coefficient,
    _lagged_cb_ym,
    _ndcg_at_k,
    _parse_amount_bin,
    _parse_ym_from_filename,
    _read_csv_selected,
    _safe_col,
    _to_numeric,
    _ym_to_quarter,
)
from recommender.moe_harness import MoEHarness

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
        self.product_source_lookup: Dict[str, Dict[str, Any]] = {}
        self.product_schema_info: Dict[str, Any] = {}
        self.moe_harness: Optional[MoEHarness] = MoEHarness() if self.config.use_moe_harness else None

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
            nrows = getattr(self.config, "table11_nrows_per_file", None)
            if nrows:
                try:
                    header = pd.read_csv(csv_path, nrows=0, encoding="utf-8")
                    usecols = [c for c in needed_cols if c in header.columns]
                    df = pd.read_csv(
                        csv_path,
                        usecols=usecols,
                        dtype=str,
                        nrows=int(nrows),
                        low_memory=False,
                        encoding="utf-8",
                    )
                except UnicodeDecodeError:
                    header = pd.read_csv(csv_path, nrows=0, encoding="cp949")
                    usecols = [c for c in needed_cols if c in header.columns]
                    df = pd.read_csv(
                        csv_path,
                        usecols=usecols,
                        dtype=str,
                        nrows=int(nrows),
                        low_memory=False,
                        encoding="cp949",
                    )
            else:
                df = _read_csv_selected(csv_path, needed_cols)
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
        schema_path = self._table12_path / "12금융상품정보.xlsx"

        def _read_csv_safe(path):
            try:
                # 1. 먼저 UTF-8로 시도
                return pd.read_csv(path, low_memory=False, encoding='utf-8')
            except UnicodeDecodeError:
                # 2. 실패하면 CP949로 시도
                return pd.read_csv(path, low_memory=False, encoding='cp949')

        dep = _read_csv_safe(deposit_path)
        fund = _read_csv_safe(fund_path)

        def _load_table12_schema_info(path: Path) -> Dict[str, Any]:
            out: Dict[str, Any] = {
                "loaded": False,
                "path": str(path),
                "deposit_columns": [],
                "fund_columns": [],
                "deposit_hard_constraint_columns": [],
            }
            if not path.exists():
                return out
            try:
                raw = pd.read_excel(path, header=None)
                if raw.shape[1] < 3:
                    return out
                m = raw.rename(columns={1: "table_name", 2: "column_name"}).copy()
                m = m.dropna(subset=["table_name", "column_name"])
                m["table_name"] = m["table_name"].astype(str).str.strip()
                m["column_name"] = m["column_name"].astype(str).str.strip()

                dep_cols = m[m["table_name"] == "은행수신상품"]["column_name"].tolist()
                fund_cols = m[m["table_name"] == "공모펀드상품"]["column_name"].tolist()

                hard_candidates = [
                    "가입대상고객_조건",
                    "가입제한_조건",
                    "기타_상품가입_고려사항",
                ]
                hard_cols = [c for c in hard_candidates if c in dep_cols]

                out.update(
                    {
                        "loaded": True,
                        "deposit_columns": dep_cols,
                        "fund_columns": fund_cols,
                        "deposit_hard_constraint_columns": hard_cols,
                    }
                )
                return out
            except Exception:
                return out

        self.product_schema_info = _load_table12_schema_info(schema_path)

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

        def _build_source_lookup(
            df: pd.DataFrame,
            id_col: str,
            family: str,
            preferred_cols: List[str],
        ) -> Dict[str, Dict[str, Any]]:
            keep_cols: List[str] = []
            seen = {id_col}
            for col in preferred_cols:
                if col in df.columns and col not in seen:
                    keep_cols.append(col)
                    seen.add(col)
            if id_col not in df.columns:
                return {}

            view = df[[id_col] + keep_cols].copy()
            lookup: Dict[str, Dict[str, Any]] = {}
            for record in view.to_dict(orient="records"):
                pid = str(record.get(id_col, ""))
                if not pid:
                    continue
                payload = {
                    k: _to_jsonable(v)
                    for k, v in record.items()
                    if k != id_col
                }
                payload["product_family"] = family
                lookup[pid] = payload
            return lookup

        target_flag = dep.get("가입대상고객_조건여부", "").fillna("").astype(str).str.strip()
        target_text_raw = dep.get("가입대상고객_조건", "").fillna("").astype(str)
        target_text = np.where(target_flag.isin(["있음", "Y", "y", "1"]), target_text_raw, "")

        limit_flag = dep.get("가입제한_조건여부", "").fillna("").astype(str).str.strip()
        limit_text_raw = dep.get("가입제한_조건", "").fillna("").astype(str)
        limit_text = np.where(limit_flag.isin(["있음", "Y", "y", "1"]), limit_text_raw, "")

        note_text = dep.get("기타_상품가입_고려사항", "").fillna("").astype(str)
        hard_cols = self.product_schema_info.get("deposit_hard_constraint_columns", [])
        if hard_cols:
            hard_parts = [dep.get(c, "").fillna("").astype(str) for c in hard_cols if c in dep.columns]
        else:
            hard_parts = [pd.Series(target_text, index=dep.index), pd.Series(limit_text, index=dep.index), note_text]
        if hard_parts:
            hard_text = hard_parts[0]
            for p in hard_parts[1:]:
                hard_text = (hard_text + " " + p).astype(str)
            hard_text = hard_text.str.replace(r"\s+", " ", regex=True).str.strip()
        else:
            hard_text = pd.Series("", index=dep.index)

        dep_norm = pd.DataFrame(
            {
                "product_id": dep.get("상품코드", dep.index.astype(str)).astype(str),
                "product_name": dep.get("상품명", "deposit").astype(str),
                "product_family": "deposit",
                "bank_code": dep.get("은행코드", "").fillna("").astype(str).str.strip(),
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
                "eligibility_target_text": pd.Series(target_text, index=dep.index).astype(str),
                "eligibility_limit_text": pd.Series(limit_text, index=dep.index).astype(str),
                "eligibility_note_text": note_text,
                "eligibility_hard_text": hard_text,
                "deposit_cluster_key": (
                    dep.get("상품그룹코드", "").fillna("UNK").astype(str).str.strip()
                    + "|"
                    + dep.get("예금입출금방식", "").fillna("UNK").astype(str).str.strip()
                    + "|"
                    + dep.get("만기여부", "").fillna("UNK").astype(str).str.strip()
                ),
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
        all_products["eligibility_target_text"] = all_products.get("eligibility_target_text", "").fillna("").astype(str)
        all_products["eligibility_limit_text"] = all_products.get("eligibility_limit_text", "").fillna("").astype(str)
        all_products["eligibility_note_text"] = all_products.get("eligibility_note_text", "").fillna("").astype(str)
        all_products["eligibility_hard_text"] = all_products.get("eligibility_hard_text", "").fillna("").astype(str)
        all_products["deposit_cluster_key"] = all_products.get("deposit_cluster_key", "").fillna("").astype(str)
        all_products["bank_code"] = all_products.get("bank_code", "").fillna("").astype(str)

        deposit_source_cols = [
            "은행코드",
            "은행명",
            "상품코드",
            "상품명",
            "상품일련번호",
            "상품그룹코드",
            "상품그룹명",
            "예금입출금방식",
            "만기여부",
            "이자지급방법",
            "이자계산방법",
            "가입대상고객_조건",
            "가입제한_조건",
            "기본금리",
            "최대우대금리",
            "우대금리조건_개수",
            "예금자보호대상여부",
            "세제혜택_비과세종합저축_여부",
            "가입금액_최소구간",
            "가입금액_최대구간",
            "계약기간개월수_최소구간",
            "계약기간개월수_최대구간",
            "신규채널",
            "해지채널",
            "상품개요_설명",
            "기타_상품가입_고려사항",
        ]
        fund_source_cols = [
            "평가기준일",
            "펀드코드",
            "펀드명",
            "설정일",
            "운용사코드",
            "운용사명",
            "대유형",
            "중유형",
            "소유형",
            "유형BM",
            "펀드키워드",
            "투자전략",
            "설정액",
            "순자산",
            "패밀리설정액",
            "패밀리순자산",
            "1년종합등급",
            "3년종합등급",
            "5년종합등급",
            "투자위험등급",
            "판매위험등급",
            "펀드성과정보_1개월",
            "펀드성과정보_3개월",
            "펀드성과정보_6개월",
            "펀드성과정보_1년",
            "펀드표준편차_1년",
            "펀드수정샤프_1년",
            "MaximumDrawDown_1년",
            "운용보수",
            "수탁보수",
            "사무관리보수",
            "판매보수",
            "선취수수료",
            "후취수수료",
            "고난도금융상품",
            "레버리지",
            "ESG(사회책임투자형)",
            "절대수익추구",
        ]
        dep_lookup = _build_source_lookup(
            dep,
            id_col="상품코드",
            family="deposit",
            preferred_cols=deposit_source_cols,
        )
        fund_lookup = _build_source_lookup(
            fund,
            id_col="펀드코드",
            family="fund",
            preferred_cols=fund_source_cols,
        )
        self.product_source_lookup = {**dep_lookup, **fund_lookup}
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
        if self.config.enable_heuristic_id_bridge and join_rate < 0.01:
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
        overall = self.snapshot_quality_report(snapshots)

        overlap_rate = 0.0
        overlap_count = 0
        if key11 in s.columns and key09 in s.columns:
            left = set(s[key11].astype(str).unique())
            right = set(s[key09].dropna().astype(str).unique())
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
            "overall": overall,
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
    ) -> pd.DataFrame:
        risk_tol = (
            0.9 * component["complexity_tolerance"]
            + 0.5 * component["financial_activity_diversity"]
            + 0.3 * _clip01(_safe_col(df, "TOT_ASST") / 10_000_000)
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
        preference = self._build_user_preference_features(df, component)
        
        # --- TPS (Thin-Filer Potential Score) v2.1 ---
        # TPS is a supplementary user signal. It should not override suitability,
        # eligibility, or product-family specific utility.
        cb_score = _safe_col(df, "CB_SCORE", _safe_col(df, "PYE_C1M210000", 700)).fillna(700)
        overdue = _safe_col(df, "OVERDUE_CNT", _safe_col(df, "PYE_MAX_DLQ_DAY", 0)).fillna(0)
        inst_rt = _safe_col(df, "INST_CNT_RT", _safe_col(df, "PYE_C18233003", 0)).fillna(0)

        s_trust_base = (
            100.0
            - (overdue * self.config.trust_overdue_weight)
            - (inst_rt * self.config.trust_inst_weight)
        ).clip(0, 100)
        tel_consistency = component["telecom_payment_consistency"].fillna(0.5).clip(0, 1)
        s_trust = (s_trust_base * 0.8) + (tel_consistency * 20.0)
        
        # 2. Activity Score (S_Activity) - 30% (Sources: 03, 06)
        spending_amt = _safe_col(df, "TOTAL_SPENDING", _safe_col(df, "CD_USE_AMT", 0)).fillna(0)
        spending_cnt = _safe_col(df, "SPENDING_COUNT", _safe_col(df, "R3M_MBR_USE_CNT", 0)).fillna(0)
        e_pay_cnt = _safe_col(df, "PAY_VISIT_CNT", _safe_col(df, "R3M_ITRT_COMM_MESSENGER", 0)).fillna(0)
        
        if len(df) > 1:
            amt_pct = spending_amt.rank(pct=True) * 100.0
            cnt_pct = spending_cnt.rank(pct=True) * 100.0
            digi_pct = e_pay_cnt.rank(pct=True) * 100.0
            s_activity = (
                amt_pct * self.config.activity_amt_weight
                + cnt_pct * self.config.activity_cnt_weight
                + digi_pct * self.config.activity_digi_weight
            )
        else:
            s_activity = pd.Series(50.0, index=df.index)
        
        # 3. Potential Score (S_Potential) - 30% (Sources: 01, 09, 11)
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

        s_potential = (
            income_pct * self.config.potential_income_weight
            + cb_pct * self.config.potential_cb_weight
            + tel_score * self.config.potential_tel_weight
            + youth_bonus * self.config.potential_youth_weight
        )

        w = self.config.tps_weights
        tps = (
            s_trust * w.get("trust", 0.7)
            + s_activity * w.get("activity", 0.15)
            + s_potential * w.get("potential", 0.15)
        )
        
        tps_features = pd.DataFrame({
            "tps_score": tps,
            "tps_trust": s_trust,
            "tps_activity": s_activity,
            "tps_potential": s_potential
        }, index=df.index)
        
        engineered = pd.concat([component, preference, tps_features], axis=1, copy=False)
        df_clean = df.drop(columns=[c for c in engineered.columns if c in df.columns], errors="ignore")
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

    @staticmethod
    def _infer_user_age(user_row: pd.Series) -> float:
        age_raw = user_row.get("AGE", np.nan)
        age = pd.to_numeric(pd.Series([age_raw]), errors="coerce").iloc[0]
        if not pd.isna(age):
            return float(age)
        age_gb = str(user_row.get("AGE_GB", "")).strip()
        m = re.search(r"(\d{2})\s*대", age_gb)
        if m:
            return float(int(m.group(1)) + 5)
        return 30.0

    @staticmethod
    def _extract_age_bounds(text: str) -> Tuple[Optional[int], Optional[int]]:
        age_min: Optional[int] = None
        age_max: Optional[int] = None
        if not text:
            return age_min, age_max

        for n1, n2 in re.findall(r"만\s*(\d+)\s*세\s*[~\-]\s*만?\s*(\d+)\s*세", text):
            a, b = int(n1), int(n2)
            lo, hi = min(a, b), max(a, b)
            age_min = lo if age_min is None else max(age_min, lo)
            age_max = hi if age_max is None else min(age_max, hi)

        for n, op in re.findall(r"만\s*(\d+)\s*세\s*(이상|이하|미만|초과|이내)", text):
            age = int(n)
            if op == "이상":
                age_min = age if age_min is None else max(age_min, age)
            elif op == "초과":
                age_min = age + 1 if age_min is None else max(age_min, age + 1)
            elif op in {"이하", "이내"}:
                age_max = age if age_max is None else min(age_max, age)
            elif op == "미만":
                age_max = age - 1 if age_max is None else min(age_max, age - 1)
        return age_min, age_max

    @staticmethod
    def _user_flag(user_row: pd.Series, keys: Sequence[str]) -> bool:
        true_tokens = {"y", "yes", "true", "1", "예", "있음"}
        for key in keys:
            if key not in user_row:
                continue
            v = user_row.get(key)
            if isinstance(v, str):
                if v.strip().lower() in true_tokens:
                    return True
            try:
                n = float(v)
                if not math.isnan(n) and n > 0:
                    return True
            except Exception:
                continue
        return False

    def _apply_deposit_eligibility_filter(self, items: pd.DataFrame, user_row: pd.Series) -> pd.DataFrame:
        if items.empty:
            return items
        if not bool(getattr(self.config, "enable_deposit_eligibility_filter", True)):
            return items
        if "product_family" not in items.columns:
            return items

        dep_mask = items["product_family"].eq("deposit")
        if not dep_mask.any():
            return items

        age = self._infer_user_age(user_row)
        is_senior = age >= 65.0
        has_child = self._user_flag(user_row, ["HAS_CHILD", "IS_PARENT", "CHILDREN_COUNT"])
        is_military = self._user_flag(user_row, ["IS_MILITARY", "MILITARY_SERVICE"])
        is_farmer = self._user_flag(user_row, ["IS_FARMER", "IS_FISHER"])
        is_vulnerable = self._user_flag(
            user_row,
            ["IS_MICROFINANCE", "IS_WELFARE", "IS_LOW_INCOME", "IS_JOB_INCENTIVE_RECIPIENT"],
        )
        is_foreigner = self._user_flag(user_row, ["IS_FOREIGNER"])
        is_business_user = self._user_flag(user_row, ["IS_BUSINESS", "IS_SOLE_PROPRIETOR", "HAS_BUSINESS_LICENSE"])
        has_program_approval = self._user_flag(user_row, ["HAS_PROGRAM_APPROVAL", "HAS_RECOMMENDATION_DOC"])
        has_local_residency_match = self._user_flag(user_row, ["HAS_LOCAL_RESIDENCY_MATCH", "IS_LOCAL_RESIDENT"])

        strict_patterns = [
            (r"미소금융|서민금융진흥원|자산형성상품\s*참여\s*추천서|근로장려|기초생활|차상위", is_vulnerable),
            (r"군인|장병|군복무|병역", is_military),
            (r"농어민|농업인|어업인", is_farmer),
            (r"자녀를\s*둔\s*부모|임산부", has_child),
            (r"만\s*65\s*세\s*이상|시니어|어르신", is_senior),
            (r"가입승인|승인\s*통보|추천을\s*받은|제출서류|발급한", has_program_approval),
            (r"[가-힣]+[시군구]\s*거주|도민", has_local_residency_match),
        ]

        keep = pd.Series(True, index=items.index)
        dep = items[dep_mask]
        for idx, row in dep.iterrows():
            text = str(row.get("eligibility_hard_text", "")).strip()
            if not text:
                text = " ".join(
                    [
                        str(row.get("eligibility_target_text", "")),
                        str(row.get("eligibility_limit_text", "")),
                        str(row.get("eligibility_note_text", "")),
                    ]
                )
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue

            age_min, age_max = self._extract_age_bounds(text)
            if age_min is not None and age < age_min:
                keep.loc[idx] = False
                continue
            if age_max is not None and age > age_max:
                keep.loc[idx] = False
                continue

            # Foreign-only conditions are treated as strict when no 내국인 alternative is present.
            if re.search(r"외국인", text) and not re.search(r"내국인|국민인\s*거주자|실명의\s*개인", text):
                if not is_foreigner:
                    keep.loc[idx] = False
                    continue

            # Exclude business-only products when user business eligibility is unknown/false.
            if re.search(r"개인사업자|법인", text):
                has_personal_alternative = re.search(r"실명의\s*개인|개인고객|개인\s*\(", text) is not None
                if (not has_personal_alternative) and (not is_business_user):
                    keep.loc[idx] = False
                    continue

            for pat, allowed in strict_patterns:
                if re.search(pat, text):
                    if not allowed:
                        keep.loc[idx] = False
                    break

        filtered = items[keep].copy()
        if filtered.empty:
            return items
        return filtered

    def _diversify_ranked_topk(self, ranked: pd.DataFrame, k: int) -> pd.DataFrame:
        if ranked.empty:
            return ranked
        cluster_cap = int(max(1, getattr(self.config, "deposit_cluster_cap_topk", 1)))
        bank_cap = int(max(1, getattr(self.config, "deposit_bank_cap_topk", 1)))
        if (cluster_cap <= 0 and bank_cap <= 0) or "product_family" not in ranked.columns:
            return ranked.head(k)

        chosen: List[int] = []
        chosen_set = set()

        def pick_with_limits(use_cluster_limit: bool, use_bank_limit: bool) -> None:
            dep_cluster_count: Dict[str, int] = {}
            dep_bank_count: Dict[str, int] = {}
            for idx in ranked.index:
                if len(chosen) >= k:
                    break
                if idx in chosen_set:
                    continue
                row = ranked.loc[idx]
                if str(row.get("product_family", "")) != "deposit":
                    chosen.append(idx)
                    chosen_set.add(idx)
                    continue

                cluster_key = str(row.get("deposit_cluster_key", "")).strip()
                cluster_key = re.sub(r"\s+", "", cluster_key) or f"deposit::{row.get('product_id')}"
                bank_key = str(row.get("bank_code", "")).strip() or "UNKNOWN_BANK"

                if use_cluster_limit and dep_cluster_count.get(cluster_key, 0) >= cluster_cap:
                    continue
                if use_bank_limit and dep_bank_count.get(bank_key, 0) >= bank_cap:
                    continue

                dep_cluster_count[cluster_key] = dep_cluster_count.get(cluster_key, 0) + 1
                dep_bank_count[bank_key] = dep_bank_count.get(bank_key, 0) + 1
                chosen.append(idx)
                chosen_set.add(idx)

        # pass 1: strict cluster + bank diversity
        pick_with_limits(use_cluster_limit=True, use_bank_limit=True)
        # pass 2: keep bank diversity, relax cluster
        if len(chosen) < k:
            pick_with_limits(use_cluster_limit=False, use_bank_limit=True)
        # pass 3: final fill
        if len(chosen) < k:
            for idx in ranked.index:
                if len(chosen) >= k:
                    break
                if idx in chosen_set:
                    continue
                chosen.append(idx)
                chosen_set.add(idx)
        return ranked.loc[chosen]

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
        filtered = self._apply_deposit_eligibility_filter(filtered, user_row)
        if filtered.empty:
            filtered = self._apply_deposit_eligibility_filter(seed.copy(), user_row)

        scored = self._add_pair_features(pd.DataFrame([user_row]), filtered)
        u_id = str(user_row.get(self.config.user_key_11, "unknown"))
        scored["tie_break"] = scored["product_id"].astype(str).map(
            lambda x: (
                int(hashlib.md5(f"{x}::{u_id}".encode()).hexdigest()[:8], 16) % 1000
            )
            / 1000.0
        )
        scored["exploration_bonus"] = scored["tie_break"] * 2.0
        scored["diversity_score"] = scored["baseline_score"] + scored["exploration_bonus"]
        scored = scored.sort_values(
            ["diversity_score", "tie_break", "max_rate"],
            ascending=[False, False, False],
        )

        top = scored.head(max_candidates)[item_cols]

        if len(top) < self.config.candidate_min:
            extra_pool = products[~products["product_id"].isin(top["product_id"])].copy()
            extra_pool = self._apply_deposit_eligibility_filter(extra_pool, user_row)
            extra = extra_pool.head(
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
        pair["baseline_score"] = (
            w["risk_match"] * pair["risk_match"]
            + w["liquidity_match"] * pair["liquidity_match"]
            + w["horizon_match"] * pair["horizon_match"]
            + w.get("complexity_match", 0.0) * pair["complexity_match"]
            + w.get("digital_match", 0.0) * pair["digital_match"]
        )
        return pair

    def _add_pair_features(self, users: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        pair = self._build_user_item_pairs(users, items)
        pair = self._compute_pair_match_features(pair)
        pair = self._compute_baseline_score(pair)
        return pair

    def _build_labels(self, pair: pd.DataFrame) -> pd.Series:
        import hashlib

        uid = pair[self.config.user_key_11].iloc[0] if self.config.user_key_11 in pair.columns else "default"
        seed = int(hashlib.md5(str(uid).encode()).hexdigest()[:8], 16) % (2**32)
        rng = np.random.default_rng(seed)
        random_noise = rng.normal(0, 0.2, size=len(pair))

        interaction = (
            pair["risk_match"] * pair["horizon_match"] * 0.25
            + pair["liquidity_match"] * (1.0 - pair["risk_level"] / 4.0) * 0.2
        )
        utility = (
            0.30 * pair["risk_match"]
            + 0.20 * pair["liquidity_match"]
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

    def score_pairs(
        self,
        user_snapshot: pd.Series,
        pair: pd.DataFrame,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Unified scoring entrypoint.

        Returns:
        - scores: ndarray
        - debug: scoring metadata for diagnostics
        """
        if bool(self.config.use_moe_harness):
            if self.moe_harness is None:
                self.moe_harness = MoEHarness()
            moe_result = self.moe_harness.score_pair(self, user_snapshot, pair)
            return moe_result.final_scores, {
                "score_model": "moe_harness",
                **moe_result.to_debug_dict(),
            }

        if self.model is not None and self.feature_columns:
            scores = self.model.predict(pair[self.feature_columns].fillna(0.0))
            debug_model = "lgbm_ranker"
        else:
            scores = pair["baseline_score"].to_numpy()
            debug_model = "baseline"

        u_id = str(user_snapshot.get(self.config.user_key_11, "unknown"))
        ui_noise = pair["product_id"].astype(str).map(
            lambda pid: (
                int(hashlib.md5(f"{pid}::{u_id}".encode()).hexdigest()[:8], 16) % 100
            )
            / 1000.0
        ).to_numpy()
        scores = scores + ui_noise
        if "tps_score" in user_snapshot.index:
            try:
                scores = scores + (0.1 * (float(user_snapshot["tps_score"]) / 100.0))
            except Exception:
                pass
        return scores, {
            "score_model": debug_model,
        }

    def recommend(self, user_snapshot: pd.Series, k: Optional[int] = None) -> Dict[str, object]:
        k = k or self.config.top_k
        candidates = self.generate_candidates(user_snapshot)
        pair = self._add_pair_features(pd.DataFrame([user_snapshot]), candidates)
        scores, score_debug = self.score_pairs(user_snapshot, pair)

        pair = pair.copy()
        pair["score"] = scores
        ranked = pair.sort_values("score", ascending=False)
        ranked = self._diversify_ranked_topk(ranked, k=k)

        out: Dict[str, Any] = {
            "user_id": str(user_snapshot[self.config.user_key_11]),
            "recommendations": [
                {"product_id": str(r.product_id), "score": float(r.score)}
                for r in ranked[["product_id", "score"]].itertuples(index=False)
            ],
        }
        if self.config.use_moe_harness:
            out["score_model"] = "moe_harness"
            if self.config.moe_debug:
                out["moe_debug"] = score_debug
        else:
            out["score_model"] = score_debug.get("score_model", "baseline")
        return out

    def batch_recommend(self, snapshots: pd.DataFrame, k: Optional[int] = None) -> List[Dict[str, object]]:
        return [self.recommend(row, k=k) for _, row in snapshots.iterrows()]

    def explain_recommendation(self, user_snapshot: pd.Series, k: Optional[int] = None) -> Dict[str, object]:
        from explainer.service import GroundedExplainer

        k = k or self.config.top_k
        explainer = GroundedExplainer(self)
        return explainer.explain_top_k(user_snapshot, k=k)

    def explain_recommendation_with(
        self,
        user_snapshot: pd.Series,
        k: Optional[int] = None,
        llm_renderer: Optional[object] = None,
        fallback_to_template_on_verify_fail: bool = True,
        use_explainer_moe: bool = False,
        compliance_rules_path: Optional[Path] = None,
        explainer_moe_debug: bool = False,
    ) -> Dict[str, object]:
        from explainer.service import GroundedExplainer

        k = k or self.config.top_k
        explainer = GroundedExplainer(
            self,
            llm_renderer=llm_renderer,
            fallback_to_template_on_verify_fail=fallback_to_template_on_verify_fail,
            use_explainer_moe=use_explainer_moe,
            compliance_rules_path=compliance_rules_path,
            explainer_moe_debug=explainer_moe_debug,
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
        primary_k = int(ks[0])
        recommended_item_counts: Dict[str, int] = {}

        for _, row in eval_snapshots.iterrows():
            candidates = self.generate_candidates(row)
            pair = self._add_pair_features(pd.DataFrame([row]), candidates)
            labels = self._build_labels(pair).to_numpy(dtype=float)
            baseline = pair["baseline_score"].to_numpy(dtype=float)
            model, _ = self.score_pairs(row, pair)

            candidate_sizes.append(int(len(pair)))
            for k in ks:
                k_int = int(k)
                baseline_scores_by_k[k_int].append(_ndcg_at_k(labels, baseline, k_int))
                ranked = pair.copy()
                ranked["score"] = model
                ranked = ranked.sort_values("score", ascending=False)
                ranked = self._diversify_ranked_topk(ranked, k=k_int)

                id_to_label = dict(zip(pair["product_id"].astype(str), labels))
                rec_labels = ranked["product_id"].astype(str).map(id_to_label).fillna(0.0).to_numpy(dtype=float)
                rec_scores = np.arange(len(rec_labels), 0, -1, dtype=float)
                model_scores_by_k[k_int].append(_ndcg_at_k(rec_labels, rec_scores, k_int))
                avg_rel_at_k[k_int].append(float(rec_labels.mean()) if rec_labels.size > 0 else 0.0)

                if k_int == primary_k:
                    for item_id in ranked["product_id"].astype(str).tolist():
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
        if recommended_item_counts:
            counts = np.array(list(recommended_item_counts.values()), dtype=float)
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
