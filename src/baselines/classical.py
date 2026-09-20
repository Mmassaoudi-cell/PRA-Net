"""
Classical / boosting / simple-deep benchmark suite (master task Sec. 14):
Logistic Regression, SVM, k-NN, Random Forest, ExtraTrees, XGBoost,
LightGBM, CatBoost, HistGradientBoosting, MLP -- all operating on the
flattened (k*d) sequence representation, trained on TRAIN, model-selected
on VAL, and reported on TEST by the shared driver in train/run_benchmarks.py.
"""
from __future__ import annotations
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


def flatten(X):
    return X.reshape(X.shape[0], -1)


class SklearnProbaWrapper:
    """Uniform .fit / .predict_proba / .n_params interface for the driver."""
    def __init__(self, model, name, needs_proba_fallback=False):
        self.model = model
        self.name = name
        self.needs_proba_fallback = needs_proba_fallback

    def fit(self, X, y, X_val=None, y_val=None):
        t0 = time.time()
        self.model.fit(flatten(X), y)
        self.train_time = time.time() - t0
        return self

    def predict_proba(self, X):
        Xf = flatten(X)
        if self.needs_proba_fallback:
            scores = self.model.decision_function(Xf)
            return 1 / (1 + np.exp(-scores))
        return self.model.predict_proba(Xf)[:, 1]

    def n_params(self):
        try:
            import pickle
            return len(pickle.dumps(self.model))
        except Exception:
            return -1


def build_classical_suite(seed: int = 0):
    return {
        "LogReg": SklearnProbaWrapper(
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed), "LogReg"),
        "LinearSVM": SklearnProbaWrapper(
            LinearSVC(C=1.0, max_iter=5000, random_state=seed), "LinearSVM", needs_proba_fallback=True),
        "kNN": SklearnProbaWrapper(
            KNeighborsClassifier(n_neighbors=15, n_jobs=-1), "kNN"),
        "RandomForest": SklearnProbaWrapper(
            RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=seed), "RandomForest"),
        "ExtraTrees": SklearnProbaWrapper(
            ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=seed), "ExtraTrees"),
        "HistGB": SklearnProbaWrapper(
            HistGradientBoostingClassifier(max_iter=300, random_state=seed), "HistGB"),
        "XGBoost": SklearnProbaWrapper(
            xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               tree_method="hist", n_jobs=-1, eval_metric="logloss", random_state=seed),
            "XGBoost"),
        "LightGBM": SklearnProbaWrapper(
            lgb.LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.1,
                                n_jobs=-1, verbosity=-1, random_state=seed), "LightGBM"),
        "CatBoost": SklearnProbaWrapper(
            CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                                verbose=False, random_seed=seed), "CatBoost"),
        "MLP": SklearnProbaWrapper(
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=seed), "MLP"),
    }
