"""
당뇨병 데이터 분석 대시보드 (Streamlit)
노트북 분석과 별개로, EDA·모델링 결과를 시각화해 한눈에 보여주는 대시보드.
모든 설명 문장은 명사형 어미로 작성.

실행: streamlit run dashboard/app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import analysis as A

st.set_page_config(page_title="당뇨병 분석 대시보드", page_icon="🩺", layout="wide")

# ---- 캐시 래퍼 (무거운 계산은 1회만 수행) ----
load_data = st.cache_data(A.load_data)
eda_tables = st.cache_data(A.eda_tables)
pca_outliers = st.cache_data(A.pca_outliers)
model_benchmark = st.cache_data(A.model_benchmark)
augmentation_compare = st.cache_data(A.augmentation_compare)
final_model_results = st.cache_data(A.final_model_results)

PRIMARY = "#2E5A88"

# ----------------------------------------------------------------------
# 헤더
# ----------------------------------------------------------------------
st.title("🩺 당뇨병 데이터 분석 대시보드")
st.caption("scikit-learn 당뇨병 데이터셋의 탐색적 분석(EDA)과 회귀 모델링 결과를 시각화한 대시보드")

X, y = load_data()
df = X.copy(); df["target"] = y
df_eda = df.copy()
df_eda["성별"] = X["sex"].map({0: "그룹 A", 1: "그룹 B"})
df_eda["연령대"] = pd.cut(X["age"], bins=[0, 40, 50, 60, 120],
                        labels=["~30대", "40대", "50대", "60대+"])

# 상단 핵심 지표
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("샘플 수", f"{len(X)} 명")
k2.metric("피처 수", f"{X.shape[1]} 개")
k3.metric("나이 범위", f"{int(X['age'].min())}~{int(X['age'].max())} 세")
k4.metric("타깃 평균", f"{y.mean():.1f}")
k5.metric("타깃 범위", f"{int(y.min())}~{int(y.max())}")

tab_eda, tab_corr, tab_model, tab_adv = st.tabs(
    ["📊 EDA — 분포", "🔗 EDA — 관계·구조", "🤖 모델링", "🔬 증강·해석"])

# ======================================================================
# 탭 1: EDA 분포 (타깃·성별·나이)
# ======================================================================
with tab_eda:
    st.subheader("타깃 분포")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(px.histogram(df, x="target", nbins=30, marginal="box",
                        color_discrete_sequence=[PRIMARY],
                        title="1년 후 진행도 분포"), use_container_width=True)
    with c2:
        st.markdown("**분포 요약**")
        st.dataframe(y.describe().round(1).to_frame("target"), use_container_width=True)

    st.divider()
    st.subheader("성별·나이 분석 (정규화 해제 후 복원)")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.box(df_eda, x="성별", y="target", color="성별", points="all",
                        title="성별 그룹별 타깃 분포"), use_container_width=True)
        st.caption("sklearn이 1/2의 실제 성별을 공개하지 않아 그룹 A/B로 표기")
    with c4:
        st.plotly_chart(px.box(df_eda.dropna(subset=["연령대"]), x="연령대", y="target",
                        color="연령대", title="연령대별 타깃 분포"), use_container_width=True)

    st.plotly_chart(px.scatter(df_eda, x="age", y="target", color="성별", trendline="ols",
                    opacity=0.5, title="나이 vs 타깃 (성별 구분)"), use_container_width=True)

# ======================================================================
# 탭 2: EDA 관계·구조 (상관·VIF·MI·PCA)
# ======================================================================
with tab_corr:
    tabs_e = eda_tables()
    corr = tabs_e["corr"]

    st.subheader("상관관계")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(px.imshow(corr.round(2), text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="피처 간 상관관계"), use_container_width=True)
    with c2:
        ct = tabs_e["corr_target"]
        st.plotly_chart(px.bar(x=ct.values, y=ct.index, orientation="h",
                        color=ct.values, color_continuous_scale="RdBu_r",
                        title="타깃과의 상관계수", labels={"x": "상관", "y": "변수"}),
                        use_container_width=True)

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("다중공선성 (VIF)")
        vif = tabs_e["vif"]
        st.plotly_chart(px.bar(vif, x="VIF", y="변수", orientation="h",
                        title="VIF (10 이상이면 강한 공선성)",
                        color="VIF", color_continuous_scale="Reds"), use_container_width=True)
        st.caption("혈청 지표(s1·s2 등)의 VIF가 높음 → 정규화 모델이 유리한 근거")
    with c4:
        st.subheader("비선형 의존성 (상호정보량)")
        mi = tabs_e["mi"]
        st.plotly_chart(px.bar(mi, x="MI", y="변수", orientation="h",
                        title="상호정보량(MI)", color="MI",
                        color_continuous_scale="Blues"), use_container_width=True)

    st.divider()
    st.subheader("차원 축소·이상치 탐지")
    pdf, ev = pca_outliers()
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(px.scatter(pdf, x="PC1", y="PC2", color="target",
                        color_continuous_scale="Viridis",
                        title=f"PCA 2D (누적 설명분산 {ev:.1%})"), use_container_width=True)
    with c6:
        st.plotly_chart(px.scatter(pdf, x="PC1", y="PC2", color="판정",
                        color_discrete_map={"정상": PRIMARY, "이상치": "#D62728"},
                        title=f"IsolationForest 이상치 ({(pdf['판정']=='이상치').sum()}건)"),
                        use_container_width=True)

# ======================================================================
# 탭 3: 모델링 (성능 비교)
# ======================================================================
with tab_model:
    st.subheader("모델 성능 비교 — 원본 vs 파생 피처")
    with st.spinner("모델 학습 중... (최초 1회, 이후 캐시)"):
        cmp = model_benchmark()

    plot_df = cmp.melt(id_vars="모델", value_vars=["R²_원본", "R²_파생"],
                       var_name="피처셋", value_name="R²")
    st.plotly_chart(px.bar(plot_df, x="모델", y="R²", color="피처셋", barmode="group",
                    title="모델별 5-fold 교차검증 R²"), use_container_width=True)

    best = cmp.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("최고 모델", best["모델"])
    c2.metric("최고 R² (파생)", f"{best['R²_파생']:.3f}")
    c3.metric("파생 평균 개선폭", f"{cmp['개선폭'].mean():+.3f}")

    st.dataframe(cmp.round(4), use_container_width=True, hide_index=True)
    st.info("트리 앙상블이 항상 우월하지는 않으며, 정규화 선형 모델이 이 데이터에서 경쟁력 있음. "
            "파생변수는 모델에 따라 소폭의 개선에 기여.")

# ======================================================================
# 탭 4: 증강·해석
# ======================================================================
with tab_adv:
    st.subheader("데이터 증강 — 엄격 비교")
    st.caption("증강은 학습 폴드에만 적용하고 테스트 폴드는 항상 원본 유지 → 데이터 누수 차단")
    with st.spinner("증강 비교 계산 중..."):
        aug = augmentation_compare()
    st.plotly_chart(px.bar(aug, x="증강 방식", y="R²", error_y="표준편차",
                    color="증강 방식", title="증강 방식별 성능 (테스트는 항상 원본)"),
                    use_container_width=True)
    st.warning("표 형식 회귀에서 합성 증강은 분포를 모방할 뿐 새 정보를 만들지 못해, "
               "누수 없는 비교에서는 뚜렷한 개선이 나타나지 않음.")

    st.divider()
    st.subheader("최종 모델 해석 (튜닝된 HistGBM)")
    with st.spinner("최종 모델 학습·해석 중..."):
        fr = final_model_results()
    c1, c2 = st.columns(2)
    c1.metric("홀드아웃 R²", f"{fr['holdout_r2']:.3f}")
    c2.metric("홀드아웃 RMSE", f"{fr['holdout_rmse']:.1f}")

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.bar(fr["importance"], x="중요도", y="변수", orientation="h",
                        title="순열 중요도", color="중요도",
                        color_continuous_scale="Teal"), use_container_width=True)
    with c4:
        pa = fr["pred_actual"]
        fig = px.scatter(pa, x="실제값", y="예측값", opacity=0.6, title="예측 대 실제")
        lim = [pa["실제값"].min(), pa["실제값"].max()]
        fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                      line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        rs = fr["residual"]
        fig = px.scatter(rs, x="예측값", y="잔차", opacity=0.6, title="잔차 분석")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        lc = fr["learning_curve"]
        fig = go.Figure()
        fig.add_scatter(x=lc["표본수"], y=lc["학습 R²"], name="학습 R²", mode="lines+markers")
        fig.add_scatter(x=lc["표본수"], y=lc["검증 R²"], name="검증 R²", mode="lines+markers")
        fig.update_layout(title="학습곡선", xaxis_title="학습 표본 수", yaxis_title="R²")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("검증 곡선이 평탄 → 표본 수보다 피처 정보량이 성능의 병목")

st.divider()
st.caption("※ 본 대시보드는 교육용 데이터셋 기반이며 실제 임상 진단에는 사용 불가.")
