# pip install --upgrade transformers huggingface_hub
# pip install plotly
# pip install deep-translator

import streamlit as st
import pandas as pd
from transformers import pipeline
from deep_translator import GoogleTranslator
import plotly.express as px
import os
import time

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

# 0단계: 불만점에서 핵심 키워드 추출 (캐싱 적용)
@st.cache_data
def extract_complaint_keywords(complaint_text):
    """
    불만점 텍스트에서 명확한 카테고리 키워드를 추출합니다.
    예: "알약이 너무 크고 먹이기 어렵다" → ["알약 크기", "먹이기"]
    """
    if not complaint_text:
        return []
    
    # 각 카테고리별 키워드 (괄호 안은 추출되지 않아야 할 부정 표현)
    keyword_map = {
        "가격": {
            "keywords": ["비싼", "비용", "가격", "가성비", "저렴", "싸", "비싸다", "가격대", "금액"],
            "phrases": ["너무 비싸", "가격이 비", "싼 제품", "저렴한"]
        },
        "안전성": {
            "keywords": ["구토", "알레르기", "부작용", "위험", "독", "중독", "질병", "아프", "아픔", "아팠다", "병"],
            "phrases": ["구토를", "알레르기", "부작용이", "위험해", "아파"]
        },
        "편의성": {
            "keywords": ["알약", "크기", "정제", "약", "크다", "불편", "까다롭", "힘들", "어렵", "쉽", "간편"],
            "phrases": ["너무 크", "먹이기 어려", "사이즈가", "알약이 크"]
        },
        "기호성": {
            "keywords": ["맛", "쓴", "냄새", "거부", "싫어", "맛없", "향", "입맛", "거부감", "탐"],
            "phrases": ["맛이 쓴", "싫어하", "거부해", "냄새가 나"]
        },
        "영양/건강": {
            "keywords": ["효과", "영양", "건강", "성분", "비타민", "칼슘", "단백질", "효능", "개선", "좋아", "도움"],
            "phrases": ["효과가", "영양가", "성분이", "도움이"]
        }
    }
    
    keywords_found = []
    text_lower = complaint_text.lower()
    
    for category, kw_dict in keyword_map.items():
        # 정확한 문구 먼저 확인
        for phrase in kw_dict.get("phrases", []):
            if phrase in text_lower:
                keywords_found.append(f"[{category}] {phrase}")
                break  # 한 카테고리에 하나만
        
        # 문구가 없으면 키워드로 확인
        if not any(f"[{category}]" in k for k in keywords_found):
            for keyword in kw_dict.get("keywords", []):
                if keyword in text_lower:
                    keywords_found.append(f"[{category}] {keyword}")
                    break
    
    return keywords_found

# 1차: LLM(Zero-shot)으로 전처리 - 불만점 텍스트를 카테고리로 분류 (캐싱 적용)
@st.cache_data
def classify_voc_by_llm(voc_text):
    """
    Zero-shot classification으로 불만점을 카테고리로 분류
    불만점 텍스트 → LLM 분석 → 가장 높은 신뢰도의 카테고리 반환
    """
    if not voc_text:
        return None, 0.0
    
    try:
        labels = ["가격", "안전성", "편의성", "기호성", "영양/건강"]
        result = zero_shot_pipeline(voc_text, labels, multi_class=False)
        
        top_category = result['labels'][0]
        top_score = result['scores'][0]
        
        return top_category, top_score
    except Exception as e:
        print(f"LLM 분류 오류: {e}")
        return None, 0.0

# 2차: 함수로 후처리 - LLM 결과를 보완하거나 신뢰도가 낮으면 보강
def map_voc_to_category(voc_text, llm_category=None, llm_score=None):
    """
    VOC 텍스트를 카테고리로 매핑
    - 1차: LLM 분류 결과 활용 (신뢰도 > 0.7 면 그대로 사용)
    - 2차: LLM 신뢰도 낮으면 기존 함수로 보강
    """
    
    # 1차: LLM 결과가 충분히 신뢰할 만하면 그대로 사용
    if llm_category and llm_score is not None and llm_score > 0.5:
        return llm_category
    
    # 2차: 함수 기반 후처리 (키워드 매칭)
    if not voc_text:
        return None
    
    voc_lower = voc_text.lower() if voc_text else ""
    
    if any(keyword in voc_lower for keyword in ['price', 'expensive', 'cost', '가격', '비싼', '비용', '가성비', '비싸', '저렴', '할인', '행사', '창렬', '혜자', '부담', '돈']):
        return "가격"
    elif any(k in voc_lower for k in ['vomit', 'diarrhea', 'allergy', '구토', '설사', '알러지', '눈물', '부작용', '가려워', '긁어', '독해', '무서워', '위험', '이상해']):
        return "안전성"
    elif any(keyword in voc_lower for keyword in ['한입', '쏙', '작아진', '작은', '사이즈', 'size', 'small', 'tablet', '알약', '크기', '작은', '부수기', '딱딱', '편의', '가루', '날림', '급여', '간편']):
        return "편의성"
    elif any(keyword in voc_lower for keyword in ['taste', 'bitter', 'flavor', '맛', '쓴', '냄새', '잘먹', '안먹', '거부', '숨겨서', '섞어', '뱉어', '환장', '순삭']):
        return "기호성"
    elif any(keyword in voc_lower for keyword in ['health', 'nutrition', 'benefit', '영양', '건강', '성분', '효과', '기능', '함량', '개선', '변화', '눈에 띄게']):
        return "영양/건강"  
    else:
        # LLM 결과가 있으면 그걸 반환 (신뢰도 낮아도)
        return llm_category if llm_category else None

# Option A: 카피에 불만점 컨텍스트 추가
@st.cache_data
def add_voc_context(copy_text, voc_category):
    """
    카피에 불만점 카테고리 컨텍스트를 추가하여 더 정확한 분류 유도
    예: "비싸다" → "[문제 카테고리: 가격] 비싸다"
    """
    if not copy_text or not voc_category:
        return copy_text
    
    # 불만점 카테고리를 명시적으로 추가
    context_prompt = f"[불만점 카테고리: {voc_category}] {copy_text}"
    return context_prompt

# 3단계: 카피 평가 - 불만점 카테고리와의 매칭도 계산 (캐싱 적용)
@st.cache_data
def evaluate_copy_match(copy_text, voc_category, voc_text):
    """
    카피가 불만점 카테고리를 얼마나 잘 해결하는지 평가
    - 1순위: 마케터가 정의한 키워드 사전을 통한 강제 분류 (정확도 보정)
    - 2순위: LLM을 통한 문맥적 속성 분류
    - 종합 매칭 점수 반환 (0~100%)
    """
    if not voc_category or not copy_text:
        return 0.0, None
    
    try:
        # [고도화 포인트 1] 키워드 기반 카테고리 강제 추출 (LLM의 오판 방지)
        # AI에게 맡기기 전, '비싸다', '혜자', '작다' 등 명확한 키워드가 있는지 함수로 먼저 확인합니다.
        forced_category = map_voc_to_category(copy_text)
        
        # [고도화 포인트 2] AI 분석 수행 (컨텍스트 주입)
        contextualized_copy = add_voc_context(copy_text, voc_category)
        
        # AI가 단어의 의미를 더 명확히 파악하도록 서술형 레이블 사용
        labels = ["가격", "안전성", "편의성", "기호성", "영양/건강"]
        result = zero_shot_pipeline(contextualized_copy, labels, multi_class=False)
        
        # [고도화 포인트 3] 최종 속성 결정 (키워드 우선순위 적용)
        # 키워드 매칭 결과(forced_category)가 있다면 AI 결과보다 우선하여 사용합니다.
        copy_category = forced_category if forced_category else result['labels'][0]
        copy_score = result['scores'][0]
        
        # 1. 카테고리 매칭 점수 계산
        # 불만점 카테고리와 카피 속성이 일치하면 기본 점수 80점 부여
        category_match = 1.0 if copy_category == voc_category else copy_score * 0.3
        
        # 2. 키워드 매칭 점수 (불만점 키워드가 카피에 직접 포함되었는가?)
        complaint_keywords = extract_complaint_keywords(voc_text)
        keyword_matches = 0
        if complaint_keywords:
            copy_lower = copy_text.lower().replace(" ", "") # 공백 제거 후 정밀 비교
            for kw in complaint_keywords:
                keyword = kw.split("] ")[1] if "] " in kw else kw
                if keyword.lower() in copy_lower:
                    keyword_matches += 1
            keyword_score = min(keyword_matches / len(complaint_keywords), 1.0)
        else:
            keyword_score = 0.5
        
        # 3. 종합 매칭 점수 (카테고리 일치 70% + 키워드 포함 30%)
        overall_match = (category_match * 0.7 + keyword_score * 0.3) * 100
        
        return overall_match, copy_category
    
    except Exception as e:
        st.error(f"카피 평가 로직 오류: {e}")
        return 0.0, None
    
        # 3. 종합 매칭 점수 계산
        overall_match = (category_match * 0.6 + keyword_score * 0.4) * 100
        
        return overall_match, copy_category
    
    except Exception as e:
        print(f"카피 평가 오류: {e}")
        return 0.0, None

# 4단계: 강화된 승자 근거 생성 (4가지 기준 분석)
def generate_detailed_winner_reason(
    copy_a, copy_b, 
    match_score_a, match_score_b,
    copy_category_a, copy_category_b,
    res_a, res_b,
    voc_category, negative_summary
):
    """
    4가지 기준으로 상세한 승자 근거 생성:
    1. 불만점 키워드 직접 포함 여부
    2. 카테고리별 일치도 분석
    3. 점수 격차 분석
    4. 불만점별 해결책 제시
    """
    
    results = {}
    
    # 1. 불만점 키워드 포함 여부
    complaint_keywords = extract_complaint_keywords(negative_summary)
    a_keywords = [kw.split("] ")[1] if "] " in kw else kw for kw in complaint_keywords]
    
    copy_a_lower = copy_a.lower()
    copy_b_lower = copy_b.lower()
    
    a_keyword_hits = sum(1 for kw in a_keywords if kw.lower() in copy_a_lower)
    b_keyword_hits = sum(1 for kw in a_keywords if kw.lower() in copy_b_lower)
    
    results['keyword_analysis'] = {
        'a_hits': a_keyword_hits,
        'b_hits': b_keyword_hits,
        'keywords': a_keywords
    }
    
    # 2. 카테고리별 일치도
    results['category_match'] = {
        'a_match': copy_category_a == voc_category,
        'b_match': copy_category_b == voc_category,
        'a_category': copy_category_a,
        'b_category': copy_category_b,
        'voc_category': voc_category
    }
    
    # 3. 점수 격차 분석
    match_diff = abs(match_score_a - match_score_b)
    senti_diff = abs(res_a['score'] - res_b['score'])
    
    results['score_gap'] = {
        'match_diff': match_diff,
        'senti_diff': senti_diff,
        'match_gap_strength': 'significant' if match_diff >= 10 else 'marginal',
        'senti_gap_strength': 'significant' if senti_diff >= 0.1 else 'marginal'
    }
    
    # 4. 불만점별 해결책 제시
    keyword_str = " | ".join(a_keywords) if a_keywords else "특정 키워드 없음"
    results['pain_point_solution'] = {
        'pain_points': keyword_str,
        'a_addresses': a_keyword_hits > 0,
        'b_addresses': b_keyword_hits > 0
    }
    
    return results

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
        # ===== 전체 시간 측정 시작 =====
        total_start = time.time()
        print("\n" + "="*60)
        print("🔍 [분석 실행] 시작")
        print("="*60)
        
        with st.spinner('AI가 리뷰 분석 및 번역을 진행 중입니다...'):
            st.divider()
            
            # 1단계: 감성 분석
            stage1_start = time.time()
            results = []
            for r in raw_reviews:
                res = senti_pipeline(r, truncation=True, max_length=512)[0]
                results.append({"리뷰 원문": r, "감성": res['label'], "신뢰도": round(res['score'], 2)})
            stage1_time = time.time() - stage1_start
            print(f"✅ [1단계] 감성 분석 완료: {stage1_time:.2f}초")
            
            df = pd.DataFrame(results)
            st.session_state.analysis_df = df
            st.session_state.raw_reviews = raw_reviews
            
            # 2단계: 요약 생성
            stage2_start = time.time()
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
            
            stage2_time = time.time() - stage2_start
            print(f"✅ [2단계] 요약 생성 및 번역 완료: {stage2_time:.2f}초")

            # 3단계: 부정 리뷰 분석
            stage3_start = time.time()
            # 상세 데이터 및 부정 리뷰 분석
            st.subheader("📋 상세 분석 데이터")
            st.dataframe(df, use_container_width=True)
            
            negative_reviews = df[df['감성'].isin(['1 stars', '2 stars'])]
            if not negative_reviews.empty:
                low_text = " ".join(negative_reviews['리뷰 원문'].tolist())
                low_sum = summ_pipeline(low_text, max_length=40, min_length=15, truncation=True)[0]['summary_text']
                ko_low = GoogleTranslator(source='en', target='ko').translate(low_sum)
                st.session_state.negative_summary = ko_low # 세션 저장
                
                # 4단계: 불만점 키워드 추출
                stage4_start = time.time()
                # 📍 불만점에서 핵심 키워드 추출
                complaint_keywords = extract_complaint_keywords(ko_low)
                
                col_complaint1, col_complaint2 = st.columns([3, 1])
                with col_complaint1:
                    st.warning(f"⚠️ **부정적 리뷰 요약:** {ko_low}")
                with col_complaint2:
                    if complaint_keywords:
                        st.info(f"**🏷️ 불만점 키워드:**\n" + "\n".join(complaint_keywords))
                    else:
                        st.info("🏷️ 키워드를 찾을 수 없습니다")
                
                stage4_time = time.time() - stage4_start
                print(f"✅ [4단계] 불만점 키워드 추출 완료: {stage4_time:.2f}초")
                stage3_time = time.time() - stage3_start
                print(f"✅ [3단계] 부정 리뷰 분석 완료: {stage3_time:.2f}초")
            else:
                st.session_state.negative_summary = ""
                st.success("🎉 부정적 리뷰가 없습니다!")

            # ===== 전체 시간 측정 종료 =====
            total_time = time.time() - total_start
            print("\n" + "="*60)
            print(f"⏱️ [총 소요시간] {total_time:.2f}초")
            print(f"📊 [분석 완료] {len(raw_reviews)}개 리뷰 분석 완료")
            print("="*60 + "\n")
            
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

    # 불만점 키워드 표시
    if st.session_state.negative_summary:
        st.info("**💡 TIP:** 아래 키워드를 해결하는 카피가 더 효과적입니다!")
        complaint_keywords = extract_complaint_keywords(st.session_state.negative_summary)
        if complaint_keywords:
            st.write("불만점 키워드: " + " | ".join([kw.split("] ")[1] if "] " in kw else kw for kw in complaint_keywords]))
    
    col_a, col_b = st.columns(2)
    with col_a:
        copy_a = st.text_input("광고 카피 A안", placeholder="예: 소형견도 한입에 쏙!")
    with col_b:
        copy_b = st.text_input("광고 카피 B안", placeholder="예: 가성비 최고의 영양제")

    if st.button("⚖️ 카피 승자 예측"):
        if not copy_a or not copy_b:
            st.error("두 가지 카피를 모두 입력해 주세요!")
        else:
            # ===== 전체 시간 측정 시작 =====
            total_start = time.time()
            print("\n" + "="*60)
            print("⚖️ [카피 승자 예측] 시작")
            print("="*60)
            
            with st.spinner('AI가 광고 효과를 시뮬레이션 중입니다...'):
                # 1단계: 감성 분석
                stage1_start = time.time()
                res_a = senti_pipeline(copy_a.strip())[0]
                res_b = senti_pipeline(copy_b.strip())[0]
                stage1_time = time.time() - stage1_start
                print(f"✅ [1단계] A/B 감성 분석 완료: {stage1_time:.2f}초")
                
                # 2단계: 불만점 카테고리 분류
                stage2_start = time.time()
                voc_category_llm, voc_score_llm = classify_voc_by_llm(st.session_state.negative_summary)
                voc_category = map_voc_to_category(
                    st.session_state.negative_summary,
                    llm_category=voc_category_llm,
                    llm_score=voc_score_llm
                )
                if not voc_category:
                    voc_category = "전반적 만족도"
                stage2_time = time.time() - stage2_start
                print(f"✅ [2단계] 불만점 카테고리 분류 완료: {stage2_time:.2f}초")
                
                # 3단계: 각 카피의 매칭도 평가 (불만점 카테고리 기반)
                stage3_start = time.time()
                match_score_a, copy_category_a = evaluate_copy_match(
                    copy_a.strip(), 
                    voc_category, 
                    st.session_state.negative_summary
                )
                match_score_b, copy_category_b = evaluate_copy_match(
                    copy_b.strip(), 
                    voc_category, 
                    st.session_state.negative_summary
                )
                stage3_time = time.time() - stage3_start
                print(f"✅ [3단계] A/B 카피 매칭 평가 완료: {stage3_time:.2f}초")

            st.subheader("🎯 AI 카피 테스트 결과")
            
            # 📊 불만점 카테고리 표시
            st.write("**📍 불만점 카테고리:**", f"**{voc_category}**")
            st.divider()
            c1, c2 = st.columns(2)
            
            with c1:
                st.write("### **A안**")
                st.metric("긍정도", f"{res_a['label']}", f"{round(res_a['score']*100, 1)}%")
                st.metric("불만점 해결도", f"{round(match_score_a, 1)}%")
                st.write(f"**속성:** {copy_category_a}")
                
                # 해결도 상태 표시
                if match_score_a >= 60:
                    st.success(f"✅ 우수 - '{voc_category}' 카테고리를 잘 해결")
                elif match_score_a >= 40:
                    st.info(f"⚠️ 중간 - '{voc_category}' 카테고리 부분 해결")
                else:
                    st.warning(f"❌ 약함 - '{voc_category}' 카테고리와 약한 연관성")
            
            with c2:
                st.write("### **B안**")
                st.metric("긍정도", f"{res_b['label']}", f"{round(res_b['score']*100, 1)}%")
                st.metric("불만점 해결도", f"{round(match_score_b, 1)}%")
                st.write(f"**속성:** {copy_category_b}")
                
                # 해결도 상태 표시
                if match_score_b >= 60:
                    st.success(f"✅ 우수 - '{voc_category}' 카테고리를 잘 해결")
                elif match_score_b >= 40:
                    st.info(f"⚠️ 중간 - '{voc_category}' 카테고리 부분 해결")
                else:
                    st.warning(f"❌ 약함 - '{voc_category}' 카테고리와 약한 연관성")
            
            # 4단계: 상세 분석 및 승자 결정
            stage4_start = time.time()
            # 🏆 최종 승자 결정 (4가지 근거 분석)
            st.divider()
            
            # 4가지 기준으로 상세 분석
            analysis = generate_detailed_winner_reason(
                copy_a, copy_b,
                match_score_a, match_score_b,
                copy_category_a, copy_category_b,
                res_a, res_b,
                voc_category, st.session_state.negative_summary
            )
            stage4_time = time.time() - stage4_start
            print(f"✅ [4단계] 상세 분석 및 승자 결정 완료: {stage4_time:.2f}초")
            
            # 승자 결정 로직
            winner, win_reasons = None, []
            winner_votes = {}
            
            # 근거 1: 키워드 포함도 (불만점 키워드 직접 언급)
            kw_analysis = analysis['keyword_analysis']
            kw_a_count = kw_analysis['a_hits']
            kw_b_count = kw_analysis['b_hits']
            
            if kw_analysis['keywords']:
                if kw_a_count > kw_b_count:
                    win_reasons.append(f"🏷️ **키워드 포함도**: A안 {kw_a_count}개 vs B안 {kw_b_count}개 (키워드: {', '.join(kw_analysis['keywords'][:2])})")
                    winner_votes['A안'] = winner_votes.get('A안', 0) + 1
                elif kw_b_count > kw_a_count:
                    win_reasons.append(f"🏷️ **키워드 포함도**: B안 {kw_b_count}개 vs A안 {kw_a_count}개 (키워드: {', '.join(kw_analysis['keywords'][:2])})")
                    winner_votes['B안'] = winner_votes.get('B안', 0) + 1
                else:
                    win_reasons.append(f"🏷️ **키워드 포함도**: 동등 (A안 {kw_a_count}개 vs B안 {kw_b_count}개)")
            
            # 근거 2: 카테고리 일치도 (속성이 불만점과 정확히 일치?)
            cat_analysis = analysis['category_match']
            if cat_analysis['a_match'] and not cat_analysis['b_match']:
                win_reasons.append(f"🎯 **카테고리 정확도**: A안이 '{voc_category}' 속성 정확히 분류 (A: {cat_analysis['a_category']} vs B: {cat_analysis['b_category']})")
                winner_votes['A안'] = winner_votes.get('A안', 0) + 1
            elif cat_analysis['b_match'] and not cat_analysis['a_match']:
                win_reasons.append(f"🎯 **카테고리 정확도**: B안이 '{voc_category}' 속성 정확히 분류 (B: {cat_analysis['b_category']} vs A: {cat_analysis['a_category']})")
                winner_votes['B안'] = winner_votes.get('B안', 0) + 1
            else:
                win_reasons.append(f"🎯 **카테고리 정확도**: 동등 (A안: {cat_analysis['a_category']} vs B안: {cat_analysis['b_category']})")
            
            # 근거 3: 점수 격차 분석 (불만점 해결도 차이)
            gap_analysis = analysis['score_gap']
            if gap_analysis['match_gap_strength'] == 'significant':
                if match_score_a > match_score_b:
                    win_reasons.append(f"📊 **해결도 격차** (유의미): A안이 {round(gap_analysis['match_diff'], 1)}% 포인트 우위 (A: {round(match_score_a, 1)}% vs B: {round(match_score_b, 1)}%)")
                    winner_votes['A안'] = winner_votes.get('A안', 0) + 1
                else:
                    win_reasons.append(f"📊 **해결도 격차** (유의미): B안이 {round(gap_analysis['match_diff'], 1)}% 포인트 우위 (B: {round(match_score_b, 1)}% vs A: {round(match_score_a, 1)}%)")
                    winner_votes['B안'] = winner_votes.get('B안', 0) + 1
            else:
                win_reasons.append(f"📊 **해결도 격차** (근소): {round(gap_analysis['match_diff'], 1)}% 차이만 존재 (A: {round(match_score_a, 1)}% vs B: {round(match_score_b, 1)}%)")
            
            # 근거 4: 불만점별 해결책 제시
            pp_analysis = analysis['pain_point_solution']
            if pp_analysis['a_addresses'] and not pp_analysis['b_addresses']:
                win_reasons.append(f"💡 **불만점 직접 해결**: A안만 고객 불만을 명시적으로 해결 (불만점: {pp_analysis['pain_points']})")
                winner_votes['A안'] = winner_votes.get('A안', 0) + 1
            elif pp_analysis['b_addresses'] and not pp_analysis['a_addresses']:
                win_reasons.append(f"💡 **불만점 직접 해결**: B안만 고객 불만을 명시적으로 해결 (불만점: {pp_analysis['pain_points']})")
                winner_votes['B안'] = winner_votes.get('B안', 0) + 1
            else:
                a_status = "포함" if pp_analysis['a_addresses'] else "미포함"
                b_status = "포함" if pp_analysis['b_addresses'] else "미포함"
                win_reasons.append(f"💡 **불만점 직접 해결**: 동등 (A안: {a_status}, B안: {b_status}) | 불만점: {pp_analysis['pain_points']}")
            
            # 근거 5: 감성 점수 (긍정도)
            senti_diff = abs(res_a['score'] - res_b['score'])
            if senti_diff >= 0.1:  # 10% 이상 차이
                if res_a['score'] > res_b['score']:
                    win_reasons.append(f"⚖️ **감성 점수** (유의미): A안이 {round(senti_diff*100, 1)}% 포인트 우위 (A: {round(res_a['score']*100, 1)}% vs B: {round(res_b['score']*100, 1)}%)")
                    winner_votes['A안'] = winner_votes.get('A안', 0) + 1
                else:
                    win_reasons.append(f"⚖️ **감성 점수** (유의미): B안이 {round(senti_diff*100, 1)}% 포인트 우위 (B: {round(res_b['score']*100, 1)}% vs A: {round(res_a['score']*100, 1)}%)")
                    winner_votes['B안'] = winner_votes.get('B안', 0) + 1
            else:
                if res_a['score'] > res_b['score']:
                    win_reasons.append(f"⚖️ **감성 점수** (근소): A안이 {round(senti_diff*100, 1)}% 포인트 우위 (A: {round(res_a['score']*100, 1)}% vs B: {round(res_b['score']*100, 1)}%)")
                    winner_votes['A안'] = winner_votes.get('A안', 0) + 1
                else:
                    win_reasons.append(f"⚖️ **감성 점수** (근소): B안이 {round(senti_diff*100, 1)}% 포인트 우위 (B: {round(res_b['score']*100, 1)}% vs A: {round(res_a['score']*100, 1)}%)")
                    winner_votes['B안'] = winner_votes.get('B안', 0) + 1
            
            # 최종 승자 결정
            if winner_votes:
                winner = max(winner_votes, key=winner_votes.get)
                vote_count = winner_votes[winner]
            else:
                winner = "A안"
                vote_count = 0
            
            # 결과 표시
            st.success(f"🏆 **최종 승자: {winner}** ({vote_count}/5가지 근거 우위)")
            st.divider()
            
            st.write("### 📋 5가지 상세 승리 근거")
            for i, reason in enumerate(win_reasons, 1):
                st.write(f"**{i}.** {reason}")
            
            # ===== 전체 시간 측정 종료 =====
            total_time = time.time() - total_start
            print("\n" + "="*60)
            print(f"⏱️ [총 소요시간] {total_time:.2f}초")
            print(f"📊 [카피 승자] {winner}")
            print("="*60 + "\n")

else:
    st.warning("⚠️ 먼저 위에서 리뷰 분석을 완료해주세요!")