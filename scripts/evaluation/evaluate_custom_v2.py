import argparse
import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 현재 파일 위치: scripts/evaluation/evaluate_custom_v2.py
# 루트 위치: scripts/evaluation/ 의 2단계 위 폴더
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

from scripts.common.runtime_config import parse_args_with_config
from thin_filer.pipeline_config import RecommenderConfig
from thin_filer.recommender import ThinFilerRecommender

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

    # 1. 원본 데이터 로드
    print(f"Loading custom CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path, encoding="utf-8")
    
    # [수정] 모든 숫자형 컬럼의 NaN을 0으로 미리 채움 (에러 방지 핵심)
    for col in df.columns:
        if col != "CUST_ID":
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 2. 고도화된 컬럼 매핑 및 TPS 기반 성향 도출
    df["user_id"] = df["CUST_ID"]

    # [수정] AGE_GB 대신 데이터에 있는 AGE를 직접 사용
    if "AGE" in df.columns:
        df["AGE"] = pd.to_numeric(df["AGE"], errors='coerce').fillna(30)
    else:
        df["AGE"] = 30

    # TPS 지표 추출 (tps_final_data.csv의 컬럼명에 맞춤)
    # 신뢰도 관련: CB_SCORE, OVERDUE_CNT, INST_CNT_RT
    # 활동성 관련: TOTAL_SPENDING, SPENDING_COUNT, PAY_VISIT_CNT
    # 잠재력 관련: AGE, EST_INCOME, TEL_GRADE

    s_trust = df["s_trust"] if "s_trust" in df.columns else pd.Series(50.0, index=df.index)
    s_activity = df["s_activity"] if "s_activity" in df.columns else pd.Series(50.0, index=df.index)
    s_potential = df["s_potential"] if "s_potential" in df.columns else pd.Series(50.0, index=df.index)
    
    # [수정] s_potential을 100으로 나누어 정규화 (0~100 -> 0~1)
    # 이제 유저마다 risk_tol이 0.5, 1.2, 2.8 등으로 다양해집니다!
    df["risk_tol"] = (df["CB_SCORE"] / 1000 * 1.5 + (s_potential / 100) * 1.5).clip(0, 3)
    
    df["liquidity_need"] = (2.0 - (s_trust / 100 * 1.5)).clip(0, 2)
    # 기타 지표들
    df["horizon_pref"] = 1.0
    df["complexity_tol"] = ((s_trust / 100) * 2.0).clip(0, 2)
    # [수정] fillna(0)를 추가하여 NaN 에러 방지
    df["amount_bin"] = (df["EST_INCOME"].fillna(0) / 10000000).astype(int).clip(0, 3)
    df["investment_possible"] = 1.0

    df["digital_behavior_freq"] = (s_activity / 100).clip(0, 1)
    df["credit_depth"] = (df["CB_SCORE"] / 1000).clip(0, 1)
    df["credit_recency"] = 0.8
    df["telecom_payment_consistency"] = 0.9
    df["card_usage_stability"] = (s_trust / 100).clip(0, 1)
    df["spending_vs_balance_ratio"] = 0.5
    
    df["C1M210000"] = df["CB_SCORE"]
    df["CD_USE_AMT"] = df["TOTAL_SPENDING"]
    df["TOT_ASST"] = df["EST_INCOME"]
    df["R3M_MBR_USE_CNT"] = df["SPENDING_COUNT"]
    df["B1Y_MOB_OS"] = df["TEL_GRADE"]
    
    df["anchor_ym"] = 202212
    df["as_of_date"] = "2022Q4"
    df["STDT"] = 202212
    
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
