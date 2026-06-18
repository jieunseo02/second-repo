"""
대시보드용 분석 로직 모듈.
노트북과 동일한 분석을 함수로 정리 — 순수 함수라 단독 테스트 가능.
"""
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor,
                              IsolationForest)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (KFold, cross_validate, train_test_split,
                                     RandomizedSearchCV, learning_curve)
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.mixture import GaussianMixture
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

RANDOM_STATE = 42

FEATURE_DESC = {
    "age": "나이", "sex": "성별", "bmi": "체질량지수", "bp": "평균 혈압",
    "s1": "총 콜레스테롤(TC)", "s2": "LDL", "s3": "HDL", "s4": "TC/HDL 비율",
    "s5": "혈청 중성지방 로그값(ltg)", "s6": "혈당(glu)",
}


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


def _sc(model):
    """선형·거리 기반 모델은 스케일링 파이프라인으로 감싸기."""
    return make_pipeline(StandardScaler(), model)


def build_models():
    """비교 대상 13종 회귀 모델 라인업 (선형 ~ 부스팅)."""
    return {
        "LinearRegression": _sc(LinearRegression()),
        "Ridge": _sc(Ridge(alpha=1.0)),
        "Lasso": _sc(Lasso(alpha=0.1, max_iter=10000)),
        "ElasticNet": _sc(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
        "SVR": _sc(SVR(C=100, gamma="scale")),
        "KNN": _sc(KNeighborsRegressor(n_neighbors=15)),
        "DecisionTree": DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "HistGBM": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                subsample=0.9, random_state=RANDOM_STATE, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                  random_state=RANDOM_STATE, verbose=-1),
    }


# 최종 모델 튜닝용 하이퍼파라미터 분포 (트리·부스팅 계열)
PARAM_DISTS = {
    "RandomForest": {"n_estimators": [200, 400, 600], "max_depth": [None, 4, 6, 8],
                     "max_features": ["sqrt", 0.5, 1.0], "min_samples_leaf": [1, 2, 5]},
    "ExtraTrees": {"n_estimators": [200, 400, 600], "max_depth": [None, 4, 6, 8],
                   "max_features": ["sqrt", 0.5, 1.0], "min_samples_leaf": [1, 2, 5]},
    "GradientBoosting": {"n_estimators": [100, 200, 300], "learning_rate": [0.02, 0.05, 0.1],
                         "max_depth": [2, 3, 4], "subsample": [0.8, 1.0]},
    "HistGBM": {"learning_rate": [0.02, 0.05, 0.1, 0.2], "max_depth": [None, 2, 3, 4],
                "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0.0, 0.1, 1.0],
                "min_samples_leaf": [10, 20, 30]},
    "XGBoost": {"n_estimators": [200, 400, 600], "learning_rate": [0.02, 0.05, 0.1],
                "max_depth": [2, 3, 4], "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0], "reg_lambda": [1, 5, 10]},
    "LightGBM": {"n_estimators": [200, 400, 600], "learning_rate": [0.02, 0.05, 0.1],
                 "num_leaves": [15, 31, 63], "max_depth": [-1, 3, 5],
                 "subsample": [0.8, 1.0], "reg_lambda": [0, 1, 5]},
}

# 성능 평가 지표 (다중 지표 동시 측정)
SCORING = {"R2": "r2", "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"}


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


def compare_models():
    """13종 모델을 다중 지표(R²·RMSE·MAE) 5-fold 교차검증으로 비교."""
    X, y = load_data()
    Xfe = add_features(X)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, model in build_models().items():
        r = cross_validate(model, Xfe, y, cv=cv, scoring=SCORING)
        rows.append({
            "모델": name,
            "R²": r["test_R2"].mean(),
            "R² 표준편차": r["test_R2"].std(),
            "RMSE": -r["test_RMSE"].mean(),
            "MAE": -r["test_MAE"].mean(),
        })
    tab = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    tab.insert(0, "순위", tab.index + 1)
    return tab


def final_model(best_name=None):
    """비교 결과로 최종 모델 선정 → (가능하면) 튜닝 → 홀드아웃 평가·해석."""
    X, y = load_data()
    Xfe = add_features(X)

    if best_name is None:
        best_name = compare_models().iloc[0]["모델"]

    base = build_models()[best_name]
    tuned = False
    best_params = {}
    if best_name in PARAM_DISTS:
        # 트리·부스팅 계열은 RandomizedSearchCV 로 튜닝
        search = RandomizedSearchCV(base, PARAM_DISTS[best_name], n_iter=20, cv=5,
                                    scoring="r2", random_state=RANDOM_STATE, n_jobs=-1).fit(Xfe, y)
        model = search.best_estimator_
        best_params = search.best_params_
        tuned = True
    else:
        model = base  # 선형·SVR·KNN 은 정규화/기본값을 그대로 사용

    Xtr, Xte, ytr, yte = train_test_split(Xfe, y, test_size=0.2, random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    pi = permutation_importance(model, Xte, yte, n_repeats=20, random_state=RANDOM_STATE)
    imp = pd.DataFrame({"변수": Xfe.columns, "중요도": pi.importances_mean}
                       ).sort_values("중요도", ascending=True)

    resid = pd.DataFrame({"예측값": pred, "잔차": yte.values - pred})
    pred_actual = pd.DataFrame({"실제값": yte.values, "예측값": pred})

    sizes, tr_sc, te_sc = learning_curve(
        model, Xfe, y, cv=5, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDOM_STATE)
    lc = pd.DataFrame({"표본수": sizes, "학습 R²": tr_sc.mean(1), "검증 R²": te_sc.mean(1)})

    return dict(
        best_name=best_name, tuned=tuned, best_params=best_params,
        holdout_r2=float(r2_score(yte, pred)),
        holdout_rmse=float(mean_squared_error(yte, pred) ** 0.5),
        holdout_mae=float(mean_absolute_error(yte, pred)),
        importance=imp, residual=resid, pred_actual=pred_actual, learning_curve=lc)


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
