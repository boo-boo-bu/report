import streamlit as st
import pandas as pd
from transformers import pipeline
from deep_translator import GoogleTranslator
import plotly.express as px

# 1. 파이프라인 로드
@st.cache_resource
def load_models():
    senti_model = pipeline(
        "sentiment-analysis",
        model = "nlptown/bert-base-multilingual-uncased-sentiment"
    )
    summ_model = pipeline(
    "summarization",
    model = "t5-small"
    )
    return senti_model, summ_model

senti_pipeline, summ_pipeline = load_models()


# 2. Web Page 설정
st.set_page_config(
    page_title="AI 마케팅 인사이트 대시보드", 
    layout="wide"
    ) 
st.title("🐾 AI 마케팅 인사이트 대시보드")
st.sidebar.header("데이터 설정")
data_source = st.sidebar.selectbox(
    "데이터 소스 선택", 
    ["Amazon - 강아지 영양제 리뷰", "직접 입력"]
    )

# 3. 리뷰 데이터(샘플)
amazon_review = [
    "The product works well but the tablets are way too big for my small Maltese. I have to crush them every time.",
    "Effective supplement, but my dog hates the bitter taste. It's a struggle to make him eat it.",
    "Great results for joint health, but the delivery took two weeks and the box was completely smashed.",
    "Too expensive for the amount of pills. I might look for a cheaper alternative next time.",
    "Amazing! My senior dog is walking much better now. Highly recommend for old dogs."
]

# 4. input 데이터
if data_source == "Amazon - 강아지 영양제 리뷰":
    raw_reviews = amazon_review

else:
    user_input = st.text_area("분석할 리뷰를 입력하세요: ", height=200)
    raw_reviews = [
        r.strip() for r in user_input.split('\n') if r.strip()
        ]
    
# 5. 분석 실행
if st.button("🔍 고객 목소리 분석 시작"):
    if not raw_reviews:
        st.warning("분석할 리뷰가 없습니다. 리뷰를 입력하거나 소스를 선택해 주세요.")
    
    else:    
        with st.spinner('AI가 리뷰 분석 및 번역을 진행 중입니다...'):
            st.divider()
        
            results = []
            for r in raw_reviews:
                res = senti_pipeline(r, truncation=True, max_length=512)[0]
                results.append({
                    "리뷰 원문": r, 
                    "별점": res['label'], 
                    "확신도": round(res['score'], 2)
                    })
            
            df = pd.DataFrame(results)
        
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 리뷰 감성 분포")
                fig = px.bar(
                    df['AI 판정'].value_counts().sort_index(), 
                    labels={'value':'개수', 'index':'별점'}
                    )
                st.plotly_chart(fig, use_container_width=True)
    
            with col2:
                st.subheader("💡 핵심 이슈 요약")
                all_text = " ".join(raw_reviews)
                en_summary = summ_pipeline(
                    all_text, 
                    max_length=50, 
                    min_length=20,
                    truncation=True
                    )[0]['summary_text']
                ko_summary = GoogleTranslator(source='en', target='ko').translate(en_summary)
                st.info(f"**영문 요약:** {en_summary}")
                st.success(f"**한글 요약:** {ko_summary}")

            st.subheader("📋 상세 분석 데이터")
            st.dataframe(df, use_container_width=True)

# 6. 카피 시뮬레이터
st.divider()
st.header("🎯 AI 광고 카피 A/B 테스트")
st.write("위에서 발견된 고객의 페인 포인트를 해결하는 카피를 검증해 보세요.")

col_a, col_b = st.columns(2)
with col_a:
    copy_a = st.text_input(
        "광고 카피 A안", 
        value="",
        placeholder = "소형견도 한입에 쏙! 작아진 고영양제"
        )
with col_b:
    copy_b = st.text_input(
        "광고 카피 B안",
        value="", 
        placeholder = "품질은 그대로, 가격은 낮춘 합리적인 영양제"
        )
    
# 7. 카피 승자 예측
if st.button("⚖️ 카피 승자 예측"):
    if not copy_a or not copy_b:
        st.error("두 가지 카피를 모두 입력해 주세요!")
    
    else:   
        res_a = senti_pipeline(copy_a.strip())[0]
        res_b = senti_pipeline(copy_b.strip())[0]
    
        st.subheader("AI 예측 결과")
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "A안 (예측 긍정)", 
                f"{res_a['label']}", 
                f"{round(res_a['score']*100, 1)}%"
                )
        with c2:
            st.metric(
                "B안 (예측 긍정)", 
                f"{res_b['label']}", 
                f"{round(res_b['score']*100, 1)}%"
                )
    
        st.markdown("---")
        if res_a['score'] > res_b['score']:
            st.balloons()
            st.success("🎉 AI는 **A안**의 성과가 더 높을 것으로 예측합니다!")
        elif res_b['score'] > res_a['score']:
            st.balloons()
            st.success("🎉 AI는 **B안**의 성과가 더 높을 것으로 예측합니다!")
        else:
            st.info("두 카피의 예측 점수가 동일합니다. 추가적인 테스트가 필요합니다.")