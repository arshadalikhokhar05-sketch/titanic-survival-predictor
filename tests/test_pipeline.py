import numpy as np
import pandas as pd
import pytest

from titanic.data import (FEATURES, engineer_features, extract_title, guess_title,
                          load_test, load_train, split_xy)
from titanic.model import explain_passenger, load_model, predict_proba, train_and_save


@pytest.fixture(scope="module")
def train_df():
    return load_train()


@pytest.fixture(scope="module")
def trained(tmp_path_factory, train_df):
    out = tmp_path_factory.mktemp("models")
    metrics = train_and_save(train_df, out_dir=out)
    return load_model(out / "model.joblib"), metrics


def test_extract_title():
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"
    assert extract_title("Heikkinen, Miss. Laina") == "Miss"
    assert extract_title("Mlle. whatever") == "Rare"  # malformed -> Rare
    assert extract_title("Doe, Dr. Jane") == "Rare"


def test_guess_title():
    assert guess_title("male", 5) == "Master"
    assert guess_title("male", 40) == "Mr"
    assert guess_title("female", 10) == "Miss"
    assert guess_title("female", 30, sibsp=1) == "Mrs"


def test_engineer_features(train_df):
    X, y = split_xy(train_df)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == 891
    assert (X["FamilySize"] == train_df["SibSp"] + train_df["Parch"] + 1).all()
    assert set(X["IsAlone"].unique()) <= {0, 1}


def test_engineer_features_without_name_or_cabin():
    row = pd.DataFrame([dict(Pclass=3, Sex="male", Age=22.0, SibSp=1, Parch=0, Fare=7.25, Embarked="S")])
    out = engineer_features(row)
    assert out.loc[0, "Title"] == "Mr" and out.loc[0, "HasCabin"] == 0


def test_model_beats_majority_baseline(trained, train_df):
    _, metrics = trained
    majority = 1 - train_df["Survived"].mean()
    assert metrics["test"]["accuracy"] > majority + 0.1
    assert metrics["test"]["roc_auc"] > 0.8


def test_predictions_are_probabilities_and_handle_missing(trained):
    model, _ = trained
    test = load_test()  # contains missing Age / Fare
    p = predict_proba(model, test)
    assert len(p) == len(test) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_sensible_ordering(trained):
    model, _ = trained
    base = dict(Age=30.0, SibSp=0, Parch=0, Fare=30.0, Embarked="S", HasCabin=0)
    woman = pd.DataFrame([dict(base, Pclass=1, Sex="female", Title="Miss")])
    man = pd.DataFrame([dict(base, Pclass=3, Sex="male", Title="Mr")])
    assert predict_proba(model, woman)[0] > predict_proba(model, man)[0]


def test_explain_passenger(trained, train_df):
    model, _ = trained
    row = pd.DataFrame([dict(Pclass=1, Sex="female", Age=25.0, SibSp=0, Parch=0, Fare=100.0,
                             Embarked="C", Title="Miss", HasCabin=1)])
    expl = explain_passenger(model, row, engineer_features(train_df))
    assert {"Factor", "Effect"} <= set(expl.columns)
    assert len(expl) == 8
