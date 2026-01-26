# pip install --upgrade transformers huggingface_hub
# pip install plotly
# pip install deep-translator

import streamlit as st
import pandas as pd
from transformers import pipeline
from deep_translator import GoogleTranslator
import plotly.express as px
import os

# 1. 파이프라인 로드 (캐싱 적용)
@st.cache_resource
def load_models():
    # 감성 분석
    senti_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    # 요약
    summ_model = pipeline("summarization", model="t5-small")
    # 제로샷 분류 (한국어를 더 잘 이해하는 가벼운 모델 추천)
    # facebook/bart-large-mnli
    # moritz/ko-bert-base-zero-shot 
    # pinion-claire/klue-roberta-base-zero-shot-re
    # team-lucid/ko-roberta-base-nli (논리)
    # Huffon/klue-roberta-base-nli (정확도)
    # kykim/bert-kor-base (범용성)
    # beomi/kcbert-base (댓글특화)
    # snunlp/KR-SBERT-V40K-klueNLI-aug (문장유사도)
    classifier = pipeline("zero-shot-classification", model="kykim/bert-kor-base")
    return senti_model, summ_model, classifier

senti_pipeline, summ_pipeline, zero_shot_pipeline = load_models()

# VOC 텍스트를 카테고리로 매핑하는 함수
def map_voc_to_category(voc_text):
    if not voc_text:
        return None
    
    voc_lower = voc_text.lower() if voc_text else ""
    
    if any(keyword in voc_lower for keyword in ['price', 'expensive', 'cost', '가격', '비싼', '비용', '가성비', '비싸', '저렴', '할인', '행사', '창렬', '혜자', '부담', '돈']):
        return "가격"
    elif any(k in voc_lower for k in ['vomit', 'diarrhea', 'allergy', '구토', '설사', '알러지', '눈물', '부작용', '가려워', '긁어', '독해', '무서워', '위험', '이상해']):
        return "안전성"
    elif any(keyword in voc_lower for keyword in ['size', 'small', 'tablet', '알약', '크기', '작은', '부수기', '딱딱', '편의', '가루', '날림', '급여', '간편']):
        return "편의성"
    elif any(keyword in voc_lower for keyword in ['taste', 'bitter', 'flavor', '맛', '쓴', '냄새', '잘먹', '안먹', '거부', '숨겨서', '섞어', '뱉어', '환장', '순삭']):
        return "기호성"
    elif any(keyword in voc_lower for keyword in ['health', 'nutrition', 'benefit', '영양', '건강', '성분', '효과', '기능', '함량', '개선', '변화', '눈에 띄게']):
        return "영양/건강"
    else:
        return None

# 2. Web Page 설정
st.set_page_config(page_title="AI 마케팅 인사이트 대시보드", layout="wide") 
st.title("🐾 AI 마케팅 인사이트 대시보드")

# Sidebar 설정
st.sidebar.header("📊 분석 이력")
if st.sidebar.button("🔄 새 리뷰 테스트", use_container_width=True):
    st.session_state.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.header("⚙️ 데이터 설정")
data_source = st.sidebar.selectbox("데이터 소스 선택", ["Amazon - 강아지 영양제 리뷰", "직접 입력"])

# 3. 리뷰 데이터
amazon_review = [
    "The product works well but the tablets are way too big for my small Maltese. I have to crush them every time.",
    "Effective supplement, but my dog hates the bitter taste. It's a struggle to make him eat it.",
    "Great results for joint health, but the delivery took two weeks and the box was completely smashed.",
    "Too expensive for the amount of pills. I might look for a cheaper alternative next time.",
    "Amazing! My senior dog is walking much better now. Highly recommend for old dogs."
]

if data_source == "Amazon - 강아지 영양제 리뷰":
    raw_reviews = amazon_review
else:
    user_input = st.text_area("분석할 리뷰를 입력하세요 (한 줄에 하나씩):", height=200)
    raw_reviews = [r.strip() for r in user_input.split('\n') if r.strip()]

# 4. 분석 실행
if st.button("🔍 고객 목소리 분석 시작"):
    if not raw_reviews:
        st.warning("분석할 리뷰가 없습니다.")
    else:    
        with st.spinner('AI가 리뷰 분석 및 번역을 진행 중입니다...'):
            st.divider()
            results = []
            for r in raw_reviews:
                res = senti_pipeline(r, truncation=True, max_length=512)[0]
                results.append({"리뷰 원문": r, "감성": res['label'], "신뢰도": round(res['score'], 2)})
            
            df = pd.DataFrame(results)
            st.session_state.analysis_df = df
            st.session_state.raw_reviews = raw_reviews
            
            # 시각화 레이아웃
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 리뷰 감성 분포")
                fig = px.bar(df['감성'].value_counts().sort_index(), labels={'value':'개수', 'index':'감성'})
                st.plotly_chart(fig, use_container_width=True)
    
            with col2:
                st.subheader("💡 핵심 이슈 요약")
                all_text = " ".join(raw_reviews)
                en_summary = summ_pipeline(all_text, max_length=50, min_length=20, truncation=True)[0]['summary_text']
                ko_summary = GoogleTranslator(source='en', target='ko').translate(en_summary)
                st.session_state.ko_summary = ko_summary # 세션 저장
                st.info(f"**[EN]** {en_summary}")
                st.success(f"**[KO]** {ko_summary}")

            # 상세 데이터 및 부정 리뷰 분석
            st.subheader("📋 상세 분석 데이터")
            st.dataframe(df, use_container_width=True)
            
            negative_reviews = df[df['감성'].isin(['1 stars', '2 stars'])]
            if not negative_reviews.empty:
                low_text = " ".join(negative_reviews['리뷰 원문'].tolist())
                low_sum = summ_pipeline(low_text, max_length=40, min_length=15, truncation=True)[0]['summary_text']
                ko_low = GoogleTranslator(source='en', target='ko').translate(low_sum)
                st.session_state.negative_summary = ko_low # 세션 저장
                st.warning(f"⚠️ **부정적 리뷰 요약:** {ko_low}")
            else:
                st.session_state.negative_summary = ""
                st.success("🎉 부정적 리뷰가 없습니다!")

            st.session_state.analysis_completed = True

# 5. 카피 시뮬레이터
st.divider()
st.header("🎯 AI 광고 카피 A/B 테스트")

if 'analysis_completed' in st.session_state and st.session_state.analysis_completed:
    # 📋 분석 결과 요약 재표시
    st.subheader("📋 분석 결과 요약")
    col_summary1, col_summary2 = st.columns(2)
    with col_summary1:
        st.write("**💡 핵심 이슈 요약**")
        if 'ko_summary' in st.session_state:
            st.success(st.session_state.ko_summary)
    with col_summary2:
        st.write("**⚠️ 부정적 리뷰의 주요 불만점**")
        if st.session_state.negative_summary:
            st.warning(st.session_state.negative_summary)
        else:
            st.info("부정적 리뷰가 없습니다.")

    st.write("위에서 발견된 고객의 페인 포인트를 해결하는 카피를 만들어 검증해 보세요.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        copy_a = st.text_input("광고 카피 A안", placeholder="예: 소형견도 한입에 쏙!")
    with col_b:
        copy_b = st.text_input("광고 카피 B안", placeholder="예: 가성비 최고의 영양제")

    if st.button("⚖️ 카피 승자 예측"):
        if not copy_a or not copy_b:
            st.error("두 가지 카피를 모두 입력해 주세요!")
        else:
            with st.spinner('AI가 광고 효과를 시뮬레이션 중입니다...'):
                res_a = senti_pipeline(copy_a.strip())[0]
                res_b = senti_pipeline(copy_b.strip())[0]
                
                labels = ["가격", "안전성", "편의성", "기호성", "영양/건강"]
                res_zero_a = zero_shot_pipeline(copy_a.strip(), labels)
                res_zero_b = zero_shot_pipeline(copy_b.strip(), labels)
                
                top_label_a, top_score_a = res_zero_a['labels'][0], round(res_zero_a['scores'][0]*100, 1)
                top_label_b, top_score_b = res_zero_b['labels'][0], round(res_zero_b['scores'][0]*100, 1)

            st.subheader("🎯 AI 카피 테스트 결과")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("A안 (긍정도)", f"{res_a['label']}", f"{round(res_a['score']*100, 1)}%")
                st.write(f"**주요 속성:** {top_label_a} ({top_score_a}%)")
            with c2:
                st.metric("B안 (긍정도)", f"{res_b['label']}", f"{round(res_b['score']*100, 1)}%")
                st.write(f"**주요 속성:** {top_label_b} ({top_score_b}%)")

            # 승자 결정 로직
            voc_category = map_voc_to_category(st.session_state.negative_summary)
            winner, win_type = None, ""
            
            if not voc_category:
                voc_category = "전반적 만족도"

            if top_label_a == voc_category and top_label_b == voc_category:
                winner = "A안" if top_score_a >= top_score_b else "B안"
                win_type = "VOC 매칭 (비교 우위)"
            elif top_label_a == voc_category:
                winner = "A안"; win_type = "VOC 매칭 (단독)"
            elif top_label_b == voc_category:
                winner = "B안"; win_type = "VOC 매칭 (단독)"
            else:
                winner = "A안" if res_a['score'] >= res_b['score'] else "B안"
                win_type = "감성 점수 우세"

            st.success(f"🏆 **최종 승자: {winner}** ({win_type})")
            
            w_score = top_score_a if winner == "A안" else top_score_b
            w_senti = round(res_a['score']*100, 1) if winner == "A안" else round(res_b['score']*100, 1)
            
            st.info(f"**🎯 승리 근거:** '{voc_category}' 해결 지수 {w_score}% 및 긍정 확신도 {w_senti}% 기록.")

else:
    st.warning("⚠️ 먼저 위에서 리뷰 분석을 완료해주세요!")