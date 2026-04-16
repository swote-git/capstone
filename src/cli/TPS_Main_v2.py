#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]

from .runtime_config import parse_args_with_config
from common.config import RecommenderConfig
from recommender.engine import ThinFilerRecommender
from user_parser.tps import compute_tps_scores


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
    df = compute_tps_scores(pd.read_csv(args.csv_path))

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
