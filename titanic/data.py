"""Data loading and feature engineering.

All transformations here are *stateless* (they never learn anything from the
data), so they are safe to apply before the train/test split. Anything that
learns from data (imputing, scaling, encoding) lives inside the sklearn
Pipeline in ``model.py`` to avoid data leakage.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone", "HasCabin"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Survived"

_TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
}


def load_train(path: Path | str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "train.csv")


def load_test(path: Path | str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "test.csv")


def extract_title(name: str) -> str:
    """'Braund, Mr. Owen Harris' -> 'Mr'; rare titles collapse to 'Rare'."""
    try:
        raw = name.split(",")[1].split(".")[0].strip()
    except (IndexError, AttributeError):
        return "Rare"
    return _TITLE_MAP.get(raw, "Rare")


def guess_title(sex: str, age: float, sibsp: int = 0) -> str:
    """Best-guess title when no name is available (used by the GUI)."""
    if sex == "male":
        return "Master" if age < 13 else "Mr"
    if age < 18:
        return "Miss"
    return "Mrs" if sibsp > 0 else "Miss"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the model's feature columns added.

    Works on raw Kaggle-format data (needs Name / Cabin) and also on rows that
    already contain a ``Title`` / ``HasCabin`` column (e.g. from the GUI).
    """
    out = df.copy()
    out.columns = out.columns.str.strip()
    if "Title" not in out.columns:
        if "Name" in out.columns:
            out["Title"] = out["Name"].apply(extract_title)
        else:  # no names available: fall back to a sex/age based guess
            out["Title"] = [guess_title(s, a if pd.notna(a) else 30, sb)
                            for s, a, sb in zip(out["Sex"], out["Age"], out["SibSp"])]
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    if "HasCabin" not in out.columns:
        out["HasCabin"] = out["Cabin"].notna().astype(int) if "Cabin" in out.columns else 0
    out["Pclass"] = out["Pclass"].astype(str)  # treat class as a category
    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    prepared = engineer_features(df)
    return prepared[FEATURES], prepared[TARGET]
