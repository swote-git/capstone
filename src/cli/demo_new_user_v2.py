import json
from common.config import RecommenderConfig
from recommender.engine import ThinFilerRecommender
from user_parser.tps import parse_new_user

def demo():
    print("\n" + "="*60)
    print(" [ 차세대 통합 추천 엔진 - 신규 유저 추천 데모 ] ")
    print("="*60)

    # 1. 엔진 초기화
    cfg = RecommenderConfig()
    rec = ThinFilerRecommender(cfg)
    rec.load_products() # 상품 정보 로드

    # 2. 신규 유저 정보 (사용자님이 정의한 피처 명칭 사용)
    new_user = {
        "CUST_ID": "USER_NEW_999",
        "AGE": 28,
        "EST_INCOME": 35000000,
        "CB_SCORE": 750,
        "TOTAL_SPENDING": 1500000,
        "SPENDING_COUNT": 40,
        "OVERDUE_CNT": 0,
        "INST_CNT_RT": 0.1,
        "PAY_VISIT_CNT": 12,
        "TEL_GRADE": 3  # 통신등급 우수
    }

    print(f"\n[입력된 신규 유저 정보]")
    print(f" - ID: {new_user['CUST_ID']} | 나이: {new_user['AGE']} | 신용점수: {new_user['CB_SCORE']}")
    print(f" - 소득: {new_user['EST_INCOME']:,}원 | 통신등급: {new_user['TEL_GRADE']}")

    # 3. 추천 실행 (새로 만든 recommend_new_user 인터페이스 활용)
    print("\n추천 엔진 가동 중 (TPS v2.0 분석 포함)...")
    result = rec.recommend_new_user(parse_new_user(new_user), k=5)

    # 4. 결과 출력
    print("\n" + "-"*60)
    print(f" {new_user['CUST_ID']}님을 위한 맞춤형 금융 상품 TOP 5 ")
    print("-"*60)
    
    for i, r in enumerate(result["recommendations"], 1):
        # 상품 상세 정보 조회
        p_info = rec.products[rec.products["product_id"] == r["product_id"]].iloc[0]
        p_name = p_info["product_name"]
        p_family = p_info["product_family"]
        
        print(f" {i}위. [{p_family}] {p_name}")
        print(f"      (추천 점수: {r['score']:.4f})")

    print("\n" + "="*60)

if __name__ == "__main__":
    demo()
