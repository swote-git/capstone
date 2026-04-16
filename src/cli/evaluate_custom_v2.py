import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

from .runtime_config import parse_args_with_config
from common.config import RecommenderConfig
from recommender.engine import ThinFilerRecommender
from user_parser.tps import parse_custom_user_frame

def parse_args():
    p = argparse.ArgumentParser()
    # 기본 경로를 프로젝트 루트 기준으로 수정
    default_csv = ROOT_DIR / "data" / "thin_filer" / "신파일러_군집_최종_피처_통합.csv"
    p.add_argument("--csv-path", type=Path, default=default_csv)
    p.add_argument("--sample-users", type=int, default=500)
    p.add_argument("--max-train-users", type=int, default=400)
    p.add_argument("--max-eval-users", type=int, default=100)
    p.add_argument("--fit", action="store_true")
    p.add_argument("--ks", nargs="+", type=int, default=[5, 10])
    return parse_args_with_config(p, section="evaluate_custom_v2")
def main():
    args = parse_args()

    print(f"Loading custom CSV: {args.csv_path}")
    df = parse_custom_user_frame(pd.read_csv(args.csv_path, encoding="utf-8"))
    
    # 3. 데이터 분할 및 학습
    users = df["user_id"].unique()
    rng = np.random.default_rng(42)
    sampled_users = rng.choice(users, min(len(users), args.sample_users), replace=False)
    df = df[df["user_id"].isin(sampled_users)].copy()
    
    cfg = RecommenderConfig(data_root=Path("data"))
    rec = ThinFilerRecommender(cfg)
    rec.load_products()
    
    train_users = sampled_users[:args.max_train_users]
    eval_users = sampled_users[args.max_train_users:]
    
    train_df = df[df["user_id"].isin(train_users)].copy()
    eval_df = df[df["user_id"].isin(eval_users)].copy()
    
    if args.fit:
        print("Training model with TPS-enhanced features...")
        rec.fit(snapshots=train_df, max_users=args.max_train_users)
        
    # 4. 정량적 평가
    eval_results = rec.evaluate(eval_df, ks=args.ks, max_users=args.max_eval_users)
    
    # 5. 샘플 추천 결과 (실제 상품명 포함)
    print("\n" + "="*60)
    print("TPS-BASED PERSONALIZED RECOMMENDATION RESULTS (Samples)")
    print("="*60)
    
    for uid in list(eval_users)[:3]: 
        user_row = eval_df[eval_df["user_id"] == uid].iloc[0]
        res = rec.recommend(user_row, k=3)
        
        p_val = user_row.get('s_potential', 0.0)
        p_str = f"{p_val:.2f}" if isinstance(p_val, (int, float)) else str(p_val)
        
        print(f"\n[USER ID: {uid}]")
        print(f" - CB Score: {user_row['CB_SCORE']} | TPS Potential: {p_str}")
        print(f" - Risk Tolerance: {user_row['risk_tol']:.2f} (Calculated with TPS)")
        print(f" - Recommended Top 3:")
        for i, r in enumerate(res["recommendations"], 1):
            p_info = rec.products[rec.products["product_id"] == r["product_id"]].iloc[0]
            
            # [가독성 개선] 외계어 상품명 대신 알기 쉬운 가칭 부여
            raw_name = str(p_info.get("product_name", "Unknown"))
            p_type = p_info.get("product_family", "Product")
            
            # 한글 음절 조합이 너무 길면(외계어면) 가공
            if len(raw_name) > 10:
                clean_name = f"{p_type.upper()}-{r['product_id'][-4:]} 추천상품"
            else:
                clean_name = raw_name
            
            print(f"   {i}. {clean_name} ({p_type}) | Score: {r['score']:.4f}")
            
    print("\n" + "="*60)
    print("QUANTITATIVE EVALUATION REPORT")
    print("="*60)
    print(json.dumps(eval_results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
