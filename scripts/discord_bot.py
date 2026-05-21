import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import asyncio
import re

# 프로젝트 루트 경로 설정 (scripts/ 폴더의 부모 폴더)
SCRIPTS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPTS_DIR.parent
os.chdir(PROJECT_ROOT) # 작업 디렉토리를 프로젝트 루트로 변경
sys.path.append(str(PROJECT_ROOT / "src"))

try:
    from thin_filer.recommender import ThinFilerRecommender
    from thin_filer.pipeline import to_json
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

# .env 파일 로드
load_dotenv(PROJECT_ROOT / ".env")
TOKEN = os.getenv("DISCORD_TOKEN")

if not TOKEN:
    print("Error: DISCORD_TOKEN not found in .env file.")
    sys.exit(1)

# 봇 설정
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# 전역 변수로 추천기 및 샘플 데이터 관리
recommender = None
snapshots = None

@bot.event
async def on_ready():
    global recommender, snapshots
    print(f"--- 챗봇 로그린 성공: {bot.user.name} ---")
    
    # 봇 상태 메시지 설정 (사용자에게 명령어 노출)
    await bot.change_presence(activity=discord.Game(name="!추천 | !상품목록"))
    
    # 1. 모델 로드 (학습된 모델이 있으면 로드, 없으면 새로 생성)
    model_path = PROJECT_ROOT / "artifacts" / "lgbm_ranker.pkl"
    if model_path.exists():
        print(f"Loading pre-trained model from {model_path}...")
        try:
            recommender = ThinFilerRecommender.load(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}. Falling back to default.")
            recommender = ThinFilerRecommender()
    else:
        print("Pre-trained model not found. Using baseline recommender.")
        recommender = ThinFilerRecommender()
    
# 2. 데이터 및 상품 정보 로드
    print("Loading products and building user snapshots...")
    try:
        recommender.load_products()
        # 데모 매칭을 위해 10,000명의 유저 스냅샷 로드 (샘플링)
        snapshots = recommender.build_user_snapshots(sample_users=10000)
        print(f"Successfully loaded {len(snapshots)} user snapshots.")
    except Exception as e:
        print(f"Error loading data: {e}")
    
    print("--- 봇 준비 완료! ---")

# 한글 번역 사전
TRANSLATIONS = {
    # Feature Reasons
    "This product matches your low risk preference.": "이 상품은 사용자의 낮은 위험 선호도에 적합합니다.",
    "This product matches your medium risk preference.": "이 상품은 사용자의 보통 위험 선호도에 적합합니다.",
    "This product matches your high risk preference.": "이 상품은 사용자의 높은 위험 선호도에 적합합니다.",
    "This product matches your very_high risk preference.": "이 상품은 사용자의 매우 높은 위험 선호도에 적합합니다.",
    "This product matches your low liquidity need.": "이 상품은 사용자의 낮은 유동성 필요에 적합합니다.",
    "This product matches your medium liquidity need.": "이 상품은 사용자의 보통 유동성 필요에 적합합니다.",
    "This product matches your high liquidity need.": "이 상품은 사용자의 높은 유동성 필요에 적합합니다.",
    "This product complexity fits your low financial knowledge level.": "이 상품의 구조는 사용자의 낮은 금융 지식 수준에 적합합니다.",
    "This product complexity fits your medium financial knowledge level.": "이 상품의 구조는 사용자의 보통 금융 지식 수준에 적합합니다.",
    "This product complexity fits your high financial knowledge level.": "이 상품의 구조는 사용자의 높은 금융 지식 수준에 적합합니다.",
    "This product horizon is aligned with your investment time preference.": "이 상품의 운용 기간은 사용자의 투자 시간 선호도와 일치합니다.",
    "Your available amount is compatible with this product's minimum requirement.": "가용 자금이 이 상품의 최소 가입 금액 요건을 충족합니다.",
    "Your digital usage pattern is compatible with this product profile.": "사용자의 디지털 사용 패턴이 이 상품의 채널 특성과 잘 맞습니다.",
    "Model indicates positive contribution from max_rate.": "모델 분석 결과, 최고 금리 항목이 추천에 긍정적인 영향을 주었습니다.",
    "Model indicates positive contribution from digital-behavior fit.": "모델 분석 결과, 디지털 행동 패턴 적합도가 추천에 기여했습니다.",
    "Model indicates positive contribution from risk_tol.": "모델 분석 결과, 위험 감수 성향이 추천에 기여했습니다.",
    "The deposit family is a suitable type for your current profile.": "예적금 상품군이 현재 사용자님의 프로필에 가장 적합한 유형입니다.",
    "The fund family is a suitable type for your current profile.": "펀드 상품군이 현재 사용자님의 프로필에 가장 적합한 유형입니다.",

    # Warnings
    "Lower returns compared to investment products.": "투자 상품에 비해 기대 수익률이 낮을 수 있습니다.",
    "Principal value can fluctuate with market conditions.": "시장 상황에 따라 원금 평가 금액이 변동될 수 있습니다.",
    "Short-term losses are possible due to higher risk level.": "높은 위험 등급으로 인해 단기적 손실이 발생할 수 있습니다.",
    "Review fees, term conditions, and liquidity constraints before decision.": "가입 전 수수료, 약관 및 유동성 제한 조건을 반드시 확인하세요.",

    # Comparisons
    "Compared with fund: higher return potential but higher risk and principal fluctuation": "펀드와 비교 시: 수익 잠재력은 높으나 위험 및 원금 변동성이 큼",
    "Compared with deposit: more capital stability but typically lower return potential": "예적금과 비교 시: 자산 안정성은 높으나 일반적으로 기대 수익률이 낮음",
    "fund": "펀드",
    "deposit": "예적금",

    # Labels
    "low": "낮음",
    "medium": "보통",
    "high": "높음",
    "very_high": "매우 높음",
    "short": "단기 (1년 미만)",
    "mid": "중기 (1~3년)",
    "long": "장기 (3년 이상)",
}

def translate_explanation(text: str) -> str:
    # 단순 문자열 치환 방식
    translated = text
    for eng, kor in TRANSLATIONS.items():
        translated = translated.replace(eng, kor)
    
    # 정규 표현식 기반 복합 문장 번역 (Simple Summary 용)
    # Recommended [family] with [risk] risk and [liquidity] liquidity for a user with [risk_pref] risk preference and [liq_need] liquidity need.
    summary_pattern = r"Recommended (\w+) with (\w+) risk and (\w+) liquidity for a user with (\w+) risk preference and (\w+) liquidity need\."
    match = re.search(summary_pattern, translated)
    if match:
        fam, risk, liq, r_pref, l_need = match.groups()
        fam_kor = TRANSLATIONS.get(fam, fam)
        risk_kor = TRANSLATIONS.get(risk, risk)
        liq_kor = TRANSLATIONS.get(liq, liq)
        r_pref_kor = TRANSLATIONS.get(r_pref, r_pref)
        l_need_kor = TRANSLATIONS.get(l_need, l_need)
        
        kor_summary = (f"위험 선호도가 '{r_pref_kor}'이고 유동성 필요가 '{l_need_kor}'인 사용자를 위해, "
                       f"위험도가 '{risk_kor}'이고 유동성이 '{liq_kor}'인 {fam_kor} 상품을 추천합니다.")
        translated = translated.replace(match.group(0), kor_summary)

    # 섹션 헤더 번역
    translated = translated.replace("[Reason]", "[추천 사유]")
    translated = translated.replace("[Warning]", "[주의 사항]")
    translated = translated.replace("[Comparison]", "[상품 비교]")
    translated = translated.replace("[Simple Summary]", "[요약]")
    
    return translated

@bot.command(name="추천")
async def recommend(ctx):
    """금융 상품 추천 인터뷰 (!추천)"""
    global recommender, snapshots
    
    if recommender is None or snapshots is None or snapshots.empty:
        await ctx.send("⚠️ 시스템이 아직 준비 중입니다. 잠시만 기다려주세요.")
        return

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel

    try:
        # 1단계: 이름
        await ctx.send("👋 안녕하세요! 맞춤형 금융 상품 추천을 위한 인터뷰를 시작합니다.\n**1. 성함이 어떻게 되시나요?**")
        msg = await bot.wait_for('message', check=check, timeout=60.0)
        user_name = msg.content

        # 2단계: 나이와 자산
        await ctx.send(f"반갑습니다 {user_name}님! 매칭을 위해 정보를 입력해주세요.\n**2. 나이와 대략적인 총 자산(또는 연소득)을 띄어쓰기로 구분하여 입력해주세요.**\n(예: `25 30000000`)")
        msg = await bot.wait_for('message', check=check, timeout=60.0)
        age_input, asset_input = map(float, msg.content.split())

        # 3단계: 투자 성향
        embed_risk = discord.Embed(title="**3. 투자 성향을 선택해주세요.**", description="1️⃣ **안정추구형**: 원금 보존이 가장 중요해요.\n2️⃣ **위험중립형**: 적절한 수익을 위해 약간의 위험은 감수할 수 있어요.\n3️⃣ **공격투자형**: 높은 수익을 위해 적극적으로 투자하고 싶어요.", color=0x3498db)
        await ctx.send(embed=embed_risk)
        msg = await bot.wait_for('message', check=check, timeout=60.0)
        risk_choice = msg.content

        # 4단계: 투자 기간
        embed_horizon = discord.Embed(title="**4. 자금을 언제쯤 다시 사용하실 계획인가요?**", description="1️⃣ **1년 이내** (단기)\n2️⃣ **1~3년** (중기)\n3️⃣ **3년 이상** (장기)", color=0x3498db)
        await ctx.send(embed=embed_horizon)
        msg = await bot.wait_for('message', check=check, timeout=60.0)
        horizon_choice = msg.content

        # 5단계: 상품 카테고리 (수신상품, 펀드, 둘다)
        embed_category = discord.Embed(title="**5. 선호하시는 상품 종류가 있으신가요?**", description="1️⃣ **수신상품만** (예적금 등)\n2️⃣ **펀드만**\n3️⃣ **둘다 원함**", color=0x3498db)
        await ctx.send(embed=embed_category)
        msg = await bot.wait_for('message', check=check, timeout=60.0)
        category_choice = msg.content

        await ctx.send(f"🔍 **{user_name}**님과 가장 유사한 금융 프로필을 찾는 중입니다...")

        # --- 매칭 로직 (Distance 기반) ---
        # 나이와 자산을 정규화하여 가장 가까운 유저를 찾음
        target_age = age_input
        target_asset = asset_input
        
        # 간단한 유클리드 거리 (나이 차이 가중치 1, 자산 차이 가중치 로그 스케일)
        temp_df = snapshots.copy()
        # 안전한 숫자 변환
        temp_df['AGE_NUM'] = pd.to_numeric(temp_df['AGE'], errors='coerce').fillna(30)
        temp_df['ASST_NUM'] = pd.to_numeric(temp_df['TOT_ASST'], errors='coerce').fillna(0)

        temp_df['age_diff'] = (temp_df['AGE_NUM'] - target_age).abs() / 10.0
        temp_df['asset_diff'] = (np.log1p(temp_df['ASST_NUM']) - np.log1p(target_asset)).abs()
        temp_df['distance'] = temp_df['age_diff'] + temp_df['asset_diff']
        
        matched_user = temp_df.sort_values('distance').iloc[0]

        # --- 사용자 선택 반영 (Override) ---
        # 300만명 데이터의 실제 행동 패턴은 유지하되, 현재 입력한 선호도를 우선함
        matched_user = matched_user.copy() # 원본 시리즈 수정을 방지하기 위해 복사
        
        if risk_choice == "1":
            matched_user['risk_tol'] = 0.5
        elif risk_choice == "3":
            matched_user['risk_tol'] = 2.5
        else:
            matched_user['risk_tol'] = 1.5

        if horizon_choice == "1":
            matched_user['horizon_pref'] = 0  # short
            matched_user['liquidity_need'] = 2.5
        elif horizon_choice == "3":
            matched_user['horizon_pref'] = 2  # long
            matched_user['liquidity_need'] = 0.5
        else:
            matched_user['horizon_pref'] = 1  # mid
            matched_user['liquidity_need'] = 1.5

        # 추천 및 설명 생성
        # [수정] 사용자의 카테고리 선택 반영
        original_family = recommender.config.recommender_family
        if category_choice == "1":
            recommender.config.recommender_family = "deposit"
        elif category_choice == "2":
            recommender.config.recommender_family = "fund"
        else:
            recommender.config.recommender_family = "all"

        try:
            result = recommender.explain_recommendation(matched_user, k=5)
        finally:
            recommender.config.recommender_family = original_family
        recommendations = result.get("recommendations", [])
        
        if not recommendations:
            await ctx.send("❌ 추천 가능한 상품이 없습니다.")
            return

        # 결과 출력 (기존 Embed 형식 활용)
        app_gd_val = pd.to_numeric(matched_user.get('APP_GD', 1), errors='coerce')
        if np.isnan(app_gd_val): app_gd_val = 1
        
        features = {
            "매칭된 유저 ID": f"`{matched_user.get('CUST_ID', 'N/A')}`",
            "입력 연령/자산": f"{int(age_input)}세 / {int(asset_input):,}원",
            "매칭 신용 점수": f"{int(float(matched_user.get('PYE_SC0000000', 700)))}점",
            "소비 스타일": "디지털 친화적" if app_gd_val > 2 else "일반 소비",
        }
        
        feature_text = "\n".join([f"• {k}: {v}" for k, v in features.items()])

        embed = discord.Embed(
            title=f"💰 {user_name}님을 위한 금융 상품 추천 결과",
            description=(
                f"**[데이터 기반 매칭 분석]**\n"
                f"{feature_text}\n\n"
                f"**{user_name}**님의 성향과 가장 비슷한 씬파일러 데이터를 분석한 결과입니다."
            ),
            color=0x2ecc71
        )
        
        for i, item in enumerate(recommendations, 1):
            p_id = item.get("product_id", "Unknown")
            p_name = item.get("product_name", "Unknown")
            p_desc = item.get("description", "")
            explanation = item.get("rendered_explanation", "설명을 생성할 수 없습니다.")
            explanation_kor = translate_explanation(explanation)
            score = item.get("score", 0.0)
            
            field_value = f"📝 **상품 설명**: {p_desc}\n✨ **추천 사유**: {explanation_kor}\n(추천 점수: {score:.2f})"
            
            embed.add_field(
                name=f"{i}순위: {p_name} ({p_id})",
                value=field_value,
                inline=False
            )
        
        embed.set_footer(text="Powered by Thin-Filer Recommender | 유사 유저 매칭 엔진 적용")
        await ctx.send(embed=embed)

    except asyncio.TimeoutError:
        await ctx.send("⏰ 답변 시간이 초과되었습니다. 다시 `!추천`을 입력해주세요.")
    except Exception as e:
        print(f"Error during interview: {e}")
        import traceback
        traceback.print_exc()
        await ctx.send(f"❌ 오류가 발생했습니다: {e}")


@bot.command(name="상품목록")
async def list_products(ctx, category: str = "all"):
    """현재 시스템에 등록된 상품 목록 출력 (!상품목록 [all|deposit|fund])"""
    global recommender
    
    if recommender is None or recommender.products is None:
        await ctx.send("⚠️ 상품 정보를 로드하는 중입니다. 잠시 후 다시 시도해주세요.")
        return

    products = recommender.products
    if category == "deposit":
        products = products[products["product_family"] == "deposit"]
        title = "🏦 은행 수신 상품 목록"
    elif category == "fund":
        products = products[products["product_family"] == "fund"]
        title = "📈 공모 펀드 상품 목록"
    else:
        title = "💰 전체 금융 상품 목록"

    if products.empty:
        await ctx.send(f"❌ '{category}' 카테고리에 등록된 상품이 없습니다.")
        return

    # 너무 많으면 출력이 힘드므로 상위 10개만 예시로 보여줌
    top_products = products.head(10)
    
    embed = discord.Embed(title=title, description=f"현재 시스템에서 추천 가능한 상품 중 상위 {len(top_products)}개를 표시합니다.", color=0x3498db)
    
    for _, row in top_products.iterrows():
        p_id = row["product_id"]
        p_name = row["product_name"]
        family = "예적금" if row["product_family"] == "deposit" else "펀드"
        risk = ["매우낮음", "낮음", "보통", "높음"][min(int(row["risk_level"]), 3)]
        horizon = {"short": "단기", "mid": "중기", "long": "장기"}.get(row["horizon"], row["horizon"])
        
        value = f"• 유형: {family}\n• 위험도: {risk}\n• 기간: {horizon}"
        if row["product_family"] == "deposit":
            value += f"\n• 최고금리: {row['max_rate']}%"
        else:
            value += f"\n• 1년수익률: {row['max_rate']}%"
            
        embed.add_field(name=f"{p_name} ({p_id})", value=value, inline=True)

    embed.set_footer(text=f"총 {len(products)}개의 상품이 등록되어 있습니다. | !상품목록 [deposit|fund]로 필터링 가능")
    await ctx.send(embed=embed)


@bot.command(name="명령어")
async def help_cmd(ctx):
    help_text = (
        "**FinTalk 봇 명령어 안내**\n"
        "`!추천` : 대화형 인터뷰를 통해 맞춤형 금융 상품을 추천해줍니다.\n"
        "`!상품목록 [all|deposit|fund]` : 현재 시스템에 등록된 상품들을 확인합니다.\n"
        "`!명령어` : 현재 이 도움말을 보여줍니다."
    )
    await ctx.send(help_text)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # 봇이 멘션되거나, 인사를 하면 명령어 안내를 보여줌
    if bot.user.mentioned_in(message) or message.content in ["안녕", "하이", "hi", "hello", "help", "도움말"]:
        await message.channel.send(
            f"안녕하세요 {message.author.name}님! 👋\n"
            f"저는 맞춤형 금융 상품을 추천해드리는 **FinTalk**입니다.\n"
            f"아래 명령어를 입력해 보세요!\n\n"
            "🔹 `!추천` : 나에게 맞는 상품 찾기 (인터뷰)\n"
            "🔹 `!상품목록` : 전체 상품 리스트 보기\n"
            "🔹 `!명령어` : 전체 명령어 안내"
        )
    
    await bot.process_commands(message)

if __name__ == "__main__":
    bot.run(TOKEN)
