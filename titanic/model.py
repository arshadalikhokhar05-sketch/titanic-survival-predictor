"""Model pipelines, comparison, evaluation and persistence."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .data import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, ROOT,
                   engineer_features, split_xy)

MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])


def candidate_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE),
        "SVM (RBF)": CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def compare_models(X: pd.DataFrame, y: pd.Series, folds: int = 5) -> pd.DataFrame:
    """Stratified k-fold cross-validation for every candidate model."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, est in candidate_models().items():
        res = cross_validate(make_pipeline(est), X, y, cv=cv,
                             scoring=["accuracy", "f1", "roc_auc"])
        rows.append({
            "Model": name,
            "CV Accuracy": res["test_accuracy"].mean(),
            "CV Accuracy Std": res["test_accuracy"].std(),
            "CV F1": res["test_f1"].mean(),
            "CV ROC-AUC": res["test_roc_auc"].mean(),
        })
    return pd.DataFrame(rows).sort_values("CV Accuracy", ascending=False).reset_index(drop=True)


def evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y, proba)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def feature_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Model-agnostic permutation importance on raw (human-readable) columns."""
    res = permutation_importance(pipe, X, y, n_repeats=15, random_state=RANDOM_STATE,
                                 scoring="accuracy")
    return (pd.DataFrame({"Feature": X.columns, "Importance": res.importances_mean,
                          "Std": res.importances_std})
            .sort_values("Importance", ascending=False).reset_index(drop=True))


def train_and_save(train_df: pd.DataFrame, out_dir: Path = MODELS_DIR) -> dict:
    """Full training run: compare, pick best, evaluate on hold-out, save artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = split_xy(train_df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    comparison = compare_models(X_tr, y_tr)
    best_name = comparison.loc[0, "Model"]

    pipe = make_pipeline(candidate_models()[best_name]).fit(X_tr, y_tr)
    test_metrics = evaluate(pipe, X_te, y_te)
    train_metrics = evaluate(pipe, X_tr, y_tr)
    importance = feature_importance(pipe, X_te, y_te)

    # Baseline = the original notebook's approach (logistic regression, 7 raw features)
    baseline = _original_baseline(train_df)

    # Final model is refit on ALL labelled data for best real-world performance.
    final = make_pipeline(candidate_models()[best_name]).fit(X, y)

    metrics = {
        "best_model": best_name,
        "train": {k: v for k, v in train_metrics.items() if k not in ("roc_curve",)},
        "test": test_metrics,
        "baseline_original_notebook": baseline,
        "comparison": comparison.to_dict(orient="records"),
        "feature_importance": importance.to_dict(orient="records"),
        "n_train_rows": int(len(X_tr)), "n_test_rows": int(len(X_te)),
    }
    joblib.dump(final, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _original_baseline(train_df: pd.DataFrame) -> dict:
    """Re-run the notebook's original recipe on the same split for a fair comparison."""
    df = train_df.drop(columns="Cabin").copy()
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    X = df.drop(columns=["PassengerId", "Name", "Ticket", "Survived"])
    y = df["Survived"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    m = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    return {"accuracy": float(accuracy_score(y_te, m.predict(X_te))),
            "roc_auc": float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))}


def load_model(path: Path | None = None) -> Pipeline:
    return joblib.load(path or MODELS_DIR / "model.joblib")


def load_metrics(path: Path | None = None) -> dict:
    return json.loads((path or MODELS_DIR / "metrics.json").read_text())


def predict_proba(pipe: Pipeline, passengers: pd.DataFrame) -> np.ndarray:
    """Survival probability for raw passenger rows (engineers features first)."""
    return pipe.predict_proba(engineer_features(passengers)[FEATURES])[:, 1]


def explain_passenger(pipe: Pipeline, row: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """Single-passenger explanation by feature ablation.

    For each feature, swap it for the typical (median / most common) value of
    the training data and see how the probability moves. Positive = that
    feature pushed this passenger towards survival.
    """
    X = engineer_features(row)[FEATURES].copy()
    base = pipe.predict_proba(X)[:, 1][0]
    groups = {
        "Sex": ["Sex"], "Passenger class": ["Pclass"], "Age": ["Age"],
        "Fare": ["Fare"], "Title": ["Title"], "Embarked": ["Embarked"],
        "Family (SibSp/Parch/Size)": ["SibSp", "Parch", "FamilySize", "IsAlone"],
        "Cabin known": ["HasCabin"],
    }
    typical = {c: (background[c].median() if c in NUMERIC_FEATURES else background[c].mode()[0])
               for c in FEATURES}
    out = []
    for label, cols in groups.items():
        alt = X.copy()
        for c in cols:
            alt[c] = typical[c]
        out.append({"Factor": label, "Effect": base - pipe.predict_proba(alt)[:, 1][0]})
    return pd.DataFrame(out).sort_values("Effect")
