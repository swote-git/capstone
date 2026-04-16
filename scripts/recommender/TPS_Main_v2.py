#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

from scripts.common.runtime_config import parse_args_with_config
from thin_filer.pipeline_config import RecommenderConfig
from thin_filer.recommender import ThinFilerRecommender


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TPS v2.0 integrated analysis + quick recommender evaluation")
    p.add_argument(
        "--csv-path",
        type=Path,
        default=ROOT_DIR / "data" / "thin_filer" / "신파일러_군집_최종_피처_통합.csv",
    )
    p.add_argument("--data-root", type=Path, default=ROOT_DIR / "data")
    p.add_argument("--sample-users", type=int, default=100)
    p.add_argument("--max-train-users", type=int, default=80)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT_DIR / "reports" / "e2e" / "신파일러_TPS_최종_산출.csv",
    )
    return parse_args_with_config(p, section="tps_main_v2")


def run_all() -> None:
    args = parse_args()
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" [ 신파일러 TPS v2.0 통합 분석 및 성능 평가 시스템 ] ")
    print("=" * 70)

    print("\n[Step 1] CSV 데이터 기반 잠재력 점수(TPS) 산출 중...")
    df = pd.read_csv(args.csv_path)

    df["s_trust"] = (100.0 - (df["OVERDUE_CNT"] * 30.0) - (df["INST_CNT"] * 5.0)).clip(0, 100)
    df["s_activity"] = (
        df["TOTAL_SPENDING"].rank(pct=True) * 30
        + df["SPENDING_COUNT"].rank(pct=True) * 40
        + df["PAY_VISIT_CNT"].rank(pct=True) * 30
    )
    income_pct = df["EST_INCOME"].rank(pct=True) * 100.0
    cb_pct = df["CB_SCORE"].rank(pct=True) * 100.0
    tel_score = df["TEL_GRADE"] * 100.0
    youth_bonus = df["AGE_GB"].apply(lambda x: 100.0 if x in ["20대", "30대"] else 0.0)
    df["s_potential"] = income_pct * 0.2 + cb_pct * 0.2 + tel_score * 0.3 + youth_bonus * 0.3
    df["tps_score"] = (df["s_trust"] * 0.4) + (df["s_activity"] * 0.3) + (df["s_potential"] * 0.3)

    top_5 = df.sort_values("tps_score", ascending=False).head(5)
    print("\n>> TPS 상위 우량군 (Top 5):")
    print(top_5[["CUST_ID", "AGE_GB", "tps_score", "s_trust", "TEL_GRADE"]].to_string(index=False))

    print("\n[Step 2] 추천 시스템 성능 지표(NDCG) 측정 중...")
    config = RecommenderConfig(data_root=args.data_root)
    rec = ThinFilerRecommender(config)
    snapshots = rec.build_user_snapshots(sample_users=args.sample_users)
    rec.fit(snapshots=snapshots, max_users=args.max_train_users)
    eval_results = rec.evaluate(snapshots, ks=[5])
    ndcg = eval_results.get("metrics", {}).get("model_ndcg@5", 0.0)
    print(f" >> NDCG@5 Score: {ndcg:.4f}")
    print(json.dumps(eval_results, ensure_ascii=False, indent=2))

    print("\n[Step 3] 결과 파일 업데이트 중...")
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f" >> 산출 결과 저장 완료: {args.out_csv}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all()
