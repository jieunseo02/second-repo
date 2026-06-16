"""
당뇨병 데이터 분석 Streamlit 리포트 — 연습용(practice)

scikit-learn 내장 당뇨병(diabetes) 데이터셋으로 수행하는
탐색적 분석(EDA)과 회귀 모델링 실습 템플릿.

[실습 방법]
- `# TODO` 로 표시된 빈칸(______)을 직접 채워 코드 완성
- 각 TODO 옆 힌트 주석 참고
- 막히면 solution/app_solution.py 와 비교
- 모든 설명 문장은 명사형 어미로 작성

실행: streamlit run practice/app_practice.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ----------------------------------------------------------------------
# 페이지 기본 설정
# ----------------------------------------------------------------------
st.set_page_config(page_title="당뇨병 데이터 분석(연습)", page_icon="🩺", layout="wide")

FEATURE_DESC = {
    "age": "나이", "sex": "성별", "bmi": "체질량지수", "bp": "평균 혈압",
    "s1": "총 콜레스테롤(TC)", "s2": "LDL", "s3": "HDL", "s4": "TC/HDL 비율",
    "s5": "혈청 중성지방 로그값(ltg)", "s6": "혈당(glu)",
}


# ----------------------------------------------------------------------
# [TODO 1] 데이터 로드 함수 완성
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    """당뇨병 데이터를 DataFrame 형태로 반환."""
    # 힌트: load_diabetes() 호출 후 data.data / data.target / data.feature_names 활용
    data = load_diabetes()
    X = pd.DataFrame(______, columns=______)   # TODO: 피처 DataFrame 생성
    y = pd.Series(______, name="target")        # TODO: 타깃 Series 생성
    return X, y


# ----------------------------------------------------------------------
# 모델 학습 및 평가 함수 (일부 TODO)
# ----------------------------------------------------------------------
@st.cache_data
def train_models(X, y):
    """5종 회귀 모델 학습 후 성능 지표와 학습된 객체 반환."""
    # [TODO 2] 학습/테스트 분할 — 테스트 20%, random_state=42
    # 힌트: train_test_split(X, y, test_size=?, random_state=?)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=______, random_state=______)   # TODO

    # [TODO 3] 모델 딕셔너리에 Lasso 추가 (LassoCV 사용)
    # 힌트: make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-3, 1, 50), max_iter=10000))
    models = {
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))),
        "Lasso": ______,                                # TODO
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted = [], {}
    for name, model in models.items():
        # [TODO 4] 모델 학습 및 테스트셋 예측
        # 힌트: model.fit(...) 후 model.predict(...)
        model.fit(______, ______)                       # TODO
        pred = model.predict(______)                    # TODO

        cv_r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")
        rows.append({
            "모델": name,
            # [TODO 5] 테스트 R² 계산 — 힌트: r2_score(정답, 예측)
            "Test R²": ______,                          # TODO
            "CV R² 평균": cv_r2.mean(),
            "CV R² 표준편차": cv_r2.std(),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "MAE": mean_absolute_error(y_test, pred),
        })
        fitted[name] = model

    perf = pd.DataFrame(rows).sort_values("Test R²", ascending=False).reset_index(drop=True)
    return perf, fitted, (X_train, X_test, y_train, y_test)


# ----------------------------------------------------------------------
# 본문 시작
# ----------------------------------------------------------------------
st.title("🩺 당뇨병 데이터 분석 리포트 (연습용)")
st.caption("scikit-learn 내장 당뇨병(diabetes) 데이터셋으로 수행하는 탐색적 분석(EDA)과 회귀 모델링 실습")

X, y = load_data()
df = X.copy()
df["target"] = y
corr = df.corr()

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. 데이터 개요", "2. 탐색적 분석(EDA)", "3. 회귀 모델링", "4. 결론"])

# ===== 탭 1: 데이터 개요 =====
with tab1:
    st.header("데이터 개요")
    c1, c2, c3 = st.columns(3)
    # [TODO 6] 샘플 수, 피처 수 채우기 — 힌트: X.shape[0], X.shape[1]
    c1.metric("샘플 수", f"{______} 명")                 # TODO
    c2.metric("피처 수", f"{______} 개")                 # TODO
    c3.metric("타깃 평균", f"{y.mean():.1f}")

    st.markdown(
        "442명 환자의 10개 생리학적 지표와 1년 후 질병 진행도(target)로 구성된 데이터셋. "
        "모든 피처는 평균 0으로 표준화되어 제공됨.")

    st.subheader("변수 설명")
    desc_df = pd.DataFrame(
        {"변수": list(FEATURE_DESC.keys()), "의미": list(FEATURE_DESC.values())})
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

    st.subheader("요약 통계")
    # [TODO 7] 요약 통계 출력 — 힌트: df.describe()
    st.dataframe(______.round(3), use_container_width=True)   # TODO

# ===== 탭 2: EDA =====
with tab2:
    st.header("탐색적 분석(EDA)")

    st.subheader("타깃 변수 분포")
    # [TODO 8] 히스토그램 — 힌트: px.histogram(df, x="target", nbins=30, marginal="box")
    fig_hist = px.histogram(______, x=______, nbins=30, marginal="box",
                            title="1년 후 질병 진행도 분포")    # TODO
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown(
        f"평균 {y.mean():.1f}, 중앙값 {y.median():.1f}로 약간 오른쪽으로 치우친 분포. "
        "극단적 이상치 없이 비교적 고르게 분포함.")

    st.subheader("상관관계 히트맵")
    # [TODO 9] 상관 히트맵 — 힌트: px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig_corr = px.imshow(______, text_auto=True, aspect="auto",
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="피처 간 상관관계")              # TODO
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("타깃과의 상관계수")
    # [TODO 10] 타깃 열의 상관계수만 추출 후 내림차순 정렬
    # 힌트: corr["target"].drop("target").sort_values(ascending=False)
    corr_t = ______                                          # TODO
    corr_df = pd.DataFrame({
        "변수": [f"{k} ({FEATURE_DESC[k]})" for k in corr_t.index],
        "상관계수": corr_t.values.round(3)})
    st.dataframe(corr_df, use_container_width=True, hide_index=True)
    st.markdown(
        "체질량지수(bmi)와 혈청 중성지방(s5)이 가장 강한 양의 상관을 보임. "
        "HDL(s3)은 유일한 뚜렷한 음의 상관 — HDL이 높을수록 진행도가 낮음(의학적으로 타당).")

    st.subheader("상위 변수 산점도")
    top_feats = corr["target"].drop("target").abs().sort_values(ascending=False).head(4).index.tolist()
    sel = st.selectbox("변수 선택", top_feats,
                       format_func=lambda f: f"{f} ({FEATURE_DESC[f]})")
    # [TODO 11] 산점도 + 추세선 — 힌트: px.scatter(df, x=sel, y="target", trendline="ols", opacity=0.5)
    fig_sc = px.scatter(df, x=sel, y="target", trendline=______,
                        opacity=0.5, title=f"{sel} vs target")  # TODO
    st.plotly_chart(fig_sc, use_container_width=True)

# ===== 탭 3: 모델링 =====
with tab3:
    st.header("회귀 모델링")
    st.markdown(
        "학습 80% / 테스트 20% 분할 후 5종 모델 학습. "
        "별도 5-fold 교차검증 R²로 과적합 여부 점검. 선형 계열은 StandardScaler 적용.")

    perf, fitted, split = train_models(X, y)
    X_train, X_test, y_train, y_test = split

    st.subheader("성능 비교")
    st.dataframe(perf.round(3), use_container_width=True, hide_index=True)

    perf_melt = perf.melt(id_vars="모델", value_vars=["Test R²", "CV R² 평균"],
                          var_name="지표", value_name="R²")
    fig_perf = px.bar(perf_melt, x="모델", y="R²", color="지표",
                      barmode="group", title="모델별 R² 비교")
    st.plotly_chart(fig_perf, use_container_width=True)
    st.info(
        "트리 앙상블(RandomForest, GradientBoosting)이 선형 모델보다 오히려 낮은 성능. "
        "샘플 수가 442개로 적고 관계가 대체로 선형적이기 때문으로 해석됨.")

    best_name = perf.iloc[0]["모델"]

    st.subheader("변수 중요도")
    col_a, col_b = st.columns(2)
    with col_a:
        ridge = fitted["Ridge"].named_steps["ridgecv"]
        ridge_coef = pd.Series(ridge.coef_, index=X.columns).sort_values()
        fig_ridge = px.bar(x=ridge_coef.values, y=ridge_coef.index, orientation="h",
                           title="Ridge 회귀 계수", labels={"x": "계수", "y": "변수"})
        st.plotly_chart(fig_ridge, use_container_width=True)
    with col_b:
        rf = fitted["RandomForest"]
        # [TODO 12] RandomForest 변수 중요도 추출 — 힌트: rf.feature_importances_
        rf_imp = pd.Series(______, index=X.columns).sort_values()   # TODO
        fig_rf = px.bar(x=rf_imp.values, y=rf_imp.index, orientation="h",
                        title="RandomForest 변수 중요도", labels={"x": "중요도", "y": "변수"})
        st.plotly_chart(fig_rf, use_container_width=True)
    st.markdown("두 방법 모두에서 bmi와 s5가 가장 영향력 큰 변수로 일관되게 나타남.")

    st.subheader(f"예측 대 실제 — {best_name}")
    best_pred = fitted[best_name].predict(X_test)
    cmp = pd.DataFrame({"실제값": y_test.values, "예측값": best_pred})
    fig_pa = px.scatter(cmp, x="실제값", y="예측값", opacity=0.5,
                        title=f"{best_name} (Test R²={perf.iloc[0]['Test R²']:.3f})")
    lim = [y_test.min(), y_test.max()]
    fig_pa.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                     line=dict(color="red", dash="dash"))
    st.plotly_chart(fig_pa, use_container_width=True)

# ===== 탭 4: 결론 =====
with tab4:
    st.header("결론 및 시사점")
    st.markdown(
        """
- 체질량지수(bmi)와 혈청 중성지방(s5)이 진행도와 가장 강하게 연관된 핵심 인자
- HDL 콜레스테롤(s3)은 음의 상관 — 보호 인자로서의 역할이 데이터로도 확인됨
- 정규화 선형 모델(Lasso/Ridge)이 복잡한 앙상블보다 우수 — 항상 복잡한 모델이 우월하지는 않음
- 최고 모델도 분산의 약 47%만 설명 — 추가 피처나 비선형 상호작용 고려 시 개선 여지
- 해석 가능성이 중요하면 Lasso/Ridge가 변수 영향을 명확히 보여줘 권장됨
""")
    st.caption("※ 본 분석은 교육용 데이터셋 기반이며 실제 임상 진단에는 사용 불가.")
