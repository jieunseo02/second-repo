"""
대시보드용 분석 로직 모듈.
노트북과 동일한 분석을 함수로 정리 — 순수 함수라 단독 테스트 가능.
모든 설명 문장은 명사형 어미로 작성.
"""
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor,
                              IsolationForest)
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split, learning_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.mixture import GaussianMixture

RANDOM_STATE = 42

FEATURE_DESC = {
    "age": "나이", "sex": "성별", "bmi": "체질량지수", "bp": "평균 혈압",
    "s1": "총 콜레스테롤(TC)", "s2": "LDL", "s3": "HDL", "s4": "TC/HDL 비율",
    "s5": "혈청 중성지방 로그값(ltg)", "s6": "혈당(glu)",
}

# HistGBM 튜닝 결과 (노트북 RandomizedSearchCV에서 도출)
BEST_PARAMS = dict(learning_rate=0.05, max_depth=2, max_leaf_nodes=63,
                   l2_regularization=0.1, min_samples_leaf=10, random_state=RANDOM_STATE)


def load_data():
    """원본 스케일로 로드 후 sex 를 0/1 범주로 인코딩."""
    data = load_diabetes(scaled=False, as_frame=True)
    X = data.data.copy()
    y = data.target.copy()
    X["sex"] = (X["sex"] == 2.0).astype(int)
    return X, y


def add_features(X):
    """핵심 변수(bmi, s5) 기반 파생변수 생성."""
    d = X.copy()
    d["bmi_s5"] = d["bmi"] * d["s5"]
    d["bmi_bp"] = d["bmi"] * d["bp"]
    d["s5_bp"] = d["s5"] * d["bp"]
    d["tc_hdl_gap"] = d["s1"] - d["s3"]
    d["bmi_sq"] = d["bmi"] ** 2
    return d


def make_models():
    """벤치마크 대상 7종 회귀 모델."""
    return {
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNetCV(
            l1_ratio=[.1, .5, .9, 1], alphas=np.logspace(-3, 1, 30), max_iter=10000)),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50))),
        "Lasso": make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-3, 1, 50), max_iter=10000)),
        "SVR": make_pipeline(StandardScaler(), SVR(C=100, gamma="scale")),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE),
        "HistGBM": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def eda_tables():
    """상관·VIF·상호정보량 등 EDA 표 계산."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    X, y = load_data()
    df = X.copy(); df["target"] = y

    corr = df.corr()
    corr_target = corr["target"].drop("target").sort_values(ascending=False)

    Xc = add_constant(X)
    vif = pd.DataFrame({
        "변수": X.columns,
        "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])]
    }).sort_values("VIF", ascending=False).reset_index(drop=True)

    mi = pd.Series(mutual_info_regression(X, y, random_state=RANDOM_STATE),
                   index=X.columns).sort_values(ascending=False)
    mi_df = pd.DataFrame({"변수": mi.index, "MI": mi.values})

    return dict(corr=corr, corr_target=corr_target, vif=vif, mi=mi_df)


def pca_outliers():
    """PCA 2D 좌표와 IsolationForest 이상치 판정."""
    X, y = load_data()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2).fit(Xs)
    pcs = pca.transform(Xs)
    flag = IsolationForest(contamination=0.05, random_state=RANDOM_STATE).fit_predict(Xs)
    pdf = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pdf["target"] = y.values
    pdf["판정"] = np.where(flag == -1, "이상치", "정상")
    return pdf, float(pca.explained_variance_ratio_.sum())


def model_benchmark():
    """원본 vs 파생 피처 — 7종 모델 5-fold R² 비교."""
    X, y = load_data()
    Xfe = add_features(X)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def bench(Xd):
        rows = []
        for name, m in make_models().items():
            s = cross_val_score(m, Xd, y, cv=cv, scoring="r2")
            rows.append({"모델": name, "R²": s.mean(), "표준편차": s.std()})
        return pd.DataFrame(rows)

    base = bench(X).rename(columns={"R²": "R²_원본", "표준편차": "std_원본"})
    fe = bench(Xfe).rename(columns={"R²": "R²_파생", "표준편차": "std_파생"})
    cmp = base.merge(fe, on="모델")
    cmp["개선폭"] = cmp["R²_파생"] - cmp["R²_원본"]
    return cmp.sort_values("R²_파생", ascending=False).reset_index(drop=True)


def _aug_gaussian(Xtr, ytr, n_new, noise=0.5, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(Xtr), n_new)
    Xn = Xtr.values[idx] + rng.normal(0, noise, (n_new, Xtr.shape[1])) * Xtr.values.std(0)
    yn = ytr.values[idx] + rng.normal(0, noise, n_new) * ytr.values.std()
    return pd.DataFrame(Xn, columns=Xtr.columns), pd.Series(yn)


def _aug_gmm(Xtr, ytr, n_new, n_comp=8, seed=0):
    Z = np.column_stack([Xtr.values, ytr.values])
    gm = GaussianMixture(n_components=n_comp, covariance_type="full", random_state=seed).fit(Z)
    samp, _ = gm.sample(n_new)
    return pd.DataFrame(samp[:, :-1], columns=Xtr.columns), pd.Series(samp[:, -1])


def augmentation_compare():
    """증강은 학습 폴드에만, 테스트는 원본 유지 — 누수 없는 비교."""
    X, y = load_data()
    Xfe = add_features(X)

    def eval_aug(aug_fn=None, mult=1.0):
        cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
        sc = []
        for tr, te in cv.split(Xfe):
            Xtr, Xte = Xfe.iloc[tr], Xfe.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            if aug_fn is not None:
                Xa, ya = aug_fn(Xtr, ytr, int(len(Xtr) * mult))
                Xtr = pd.concat([Xtr, Xa], ignore_index=True)
                ytr = pd.concat([ytr, ya], ignore_index=True)
            m = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
            m.fit(Xtr, ytr)
            sc.append(r2_score(yte, m.predict(Xte)))
        return np.array(sc)

    spec = {
        "원본(증강 없음)": eval_aug(),
        "가우시안 +100%": eval_aug(_aug_gaussian, 1.0),
        "GMM +100%": eval_aug(_aug_gmm, 1.0),
        "GMM +300%": eval_aug(_aug_gmm, 3.0),
    }
    return pd.DataFrame([{"증강 방식": k, "R²": v.mean(), "표준편차": v.std()}
                        for k, v in spec.items()])


def final_model_results():
    """튜닝된 HistGBM 최종 학습 — 중요도·잔차·예측·학습곡선 산출."""
    X, y = load_data()
    Xfe = add_features(X)
    Xtr, Xte, ytr, yte = train_test_split(Xfe, y, test_size=0.2, random_state=RANDOM_STATE)
    best = HistGradientBoostingRegressor(**BEST_PARAMS)
    best.fit(Xtr, ytr)
    pred = best.predict(Xte)

    pi = permutation_importance(best, Xte, yte, n_repeats=20, random_state=RANDOM_STATE)
    imp = pd.DataFrame({"변수": Xfe.columns, "중요도": pi.importances_mean}
                       ).sort_values("중요도", ascending=True)

    resid = pd.DataFrame({"예측값": pred, "잔차": yte.values - pred})
    pred_actual = pd.DataFrame({"실제값": yte.values, "예측값": pred})

    sizes, tr_sc, te_sc = learning_curve(
        best, Xfe, y, cv=5, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDOM_STATE)
    lc = pd.DataFrame({"표본수": sizes, "학습 R²": tr_sc.mean(1), "검증 R²": te_sc.mean(1)})

    return dict(
        holdout_r2=float(r2_score(yte, pred)),
        holdout_rmse=float(mean_squared_error(yte, pred) ** 0.5),
        importance=imp, residual=resid, pred_actual=pred_actual, learning_curve=lc)
