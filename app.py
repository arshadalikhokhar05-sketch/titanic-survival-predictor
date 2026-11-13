"""Titanic Survival Predictor: interactive Streamlit app.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from titanic.data import (engineer_features, guess_title,
                          load_test, load_train)
from titanic.model import (MODELS_DIR, explain_passenger, load_metrics, load_model,
                           predict_proba, train_and_save)

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

BLUE, RED, GREEN = "#2E6F9E", "#C8553D", "#3A9D6B"
CLASS_LABELS = {1: "1st class", 2: "2nd class", 3: "3rd class"}
PORTS = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q"}

PRESETS = {
    "Custom": None,
    "Young man, 3rd class": dict(pclass=3, sex="male", age=22, sibsp=1, parch=0, fare=7.25, port="Southampton", cabin=False),
    "Young woman, 1st class": dict(pclass=1, sex="female", age=25, sibsp=0, parch=0, fare=100.0, port="Cherbourg", cabin=True),
    "Older man, 1st class": dict(pclass=1, sex="male", age=60, sibsp=0, parch=0, fare=80.0, port="Southampton", cabin=True),
    "Girl travelling with family, 2nd class": dict(pclass=2, sex="female", age=10, sibsp=1, parch=1, fare=30.0, port="Southampton", cabin=False),
}


# ---------------------------------------------------------------- cached resources
@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    df = load_train()
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # 2 missing values
    df["Class"] = df["Pclass"].map(CLASS_LABELS)
    df["Outcome"] = df["Survived"].map({0: "Did not survive", 1: "Survived"})
    return df


@st.cache_resource(show_spinner="Training model for the first time ...")
def get_model_and_metrics():
    if not (MODELS_DIR / "model.joblib").exists():
        train_and_save(load_train())
    return load_model(), load_metrics()


def typical_fare(pclass: int) -> float:
    return float(get_data().loc[get_data()["Pclass"] == pclass, "Fare"].median())


# ---------------------------------------------------------------- sidebar inputs
def apply_preset() -> None:
    p = PRESETS[st.session_state["preset"]]
    if p:
        for key, val in p.items():
            st.session_state[f"in_{key}"] = val


DEFAULTS = dict(pclass=3, sex="male", age=28, sibsp=0, parch=0, fare=8.05, port="Southampton", cabin=False)


def sidebar_passenger() -> dict:
    for key, val in DEFAULTS.items():
        st.session_state.setdefault(f"in_{key}", val)
    st.sidebar.header("🧍 Passenger")
    st.sidebar.selectbox("Quick example", list(PRESETS), key="preset", on_change=apply_preset)

    pclass = st.sidebar.radio("Ticket class", [1, 2, 3], horizontal=True,
                              format_func=CLASS_LABELS.get, key="in_pclass")
    sex = st.sidebar.radio("Sex", ["male", "female"], horizontal=True, key="in_sex")
    age = st.sidebar.slider("Age", 0, 80, key="in_age")
    sibsp = st.sidebar.number_input("Siblings / spouses aboard", 0, 8, key="in_sibsp")
    parch = st.sidebar.number_input("Parents / children aboard", 0, 6, key="in_parch")
    fare = st.sidebar.slider("Ticket fare (£)", 0.0, 520.0, step=0.5, key="in_fare",
                             help=f"Median fare in {CLASS_LABELS[pclass]}: £{typical_fare(pclass):.2f}")
    port = st.sidebar.selectbox("Port of embarkation", list(PORTS), key="in_port")
    cabin = st.sidebar.checkbox("Cabin number on record", key="in_cabin",
                                help="In the data, passengers with a recorded cabin were mostly 1st class.")

    with st.sidebar.expander("Advanced"):
        auto_title = guess_title(sex, age, sibsp)
        titles = ["Mr", "Mrs", "Miss", "Master", "Rare"]
        title = st.selectbox("Title (guessed from sex/age)", titles, index=titles.index(auto_title),
                             key=f"title_{auto_title}")

    return dict(Pclass=pclass, Sex=sex, Age=float(age), SibSp=int(sibsp), Parch=int(parch),
                Fare=float(fare), Embarked=PORTS[port], Title=title, HasCabin=int(cabin))


# ---------------------------------------------------------------- tab 1: predict
def tab_predict(model, passenger: dict) -> None:
    row = pd.DataFrame([passenger])
    prob = float(predict_proba(model, row)[0])
    survived = prob >= 0.5

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".0f"},
            title={"text": "Estimated survival probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": GREEN if survived else RED},
                   "steps": [{"range": [0, 50], "color": "rgba(200,85,61,.15)"},
                             {"range": [50, 100], "color": "rgba(58,157,107,.15)"}],
                   "threshold": {"line": {"color": "gray", "width": 3}, "value": 50}}))
        fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(fig, width="stretch")
        if survived:
            st.success(f"**Likely survived** ({prob:.0%} estimated chance)")
        else:
            st.error(f"**Likely did not survive** ({prob:.0%} estimated chance)")
        st.caption("A statistical estimate from 891 historical passengers, "
                   "not a statement about any real person.")

    with right:
        st.subheader("Why this prediction?")
        expl = explain_passenger(model, row, engineer_features(load_train()))
        expl["Colour"] = np.where(expl["Effect"] >= 0, "Raised survival chance", "Lowered survival chance")
        fig = px.bar(expl, x="Effect", y="Factor", orientation="h", color="Colour",
                     color_discrete_map={"Raised survival chance": GREEN, "Lowered survival chance": RED})
        fig.update_layout(height=340, margin=dict(t=10, b=10), legend_title=None,
                          xaxis_title="Change in survival probability (vs. a typical passenger)",
                          yaxis_title=None, xaxis_tickformat="+.0%")
        st.plotly_chart(fig, width="stretch")
        st.caption("Each bar shows how the probability changes if that factor were replaced by the "
                   "typical value from the data.")

    st.divider()
    st.subheader("🔬 What-if explorer")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ages = np.arange(0, 81, 2)
        sweep = pd.concat([pd.DataFrame([{**passenger, "Age": float(a),
                                          "Title": guess_title(passenger["Sex"], a, passenger["SibSp"])}])
                           for a in ages])
        sweep_df = pd.DataFrame({"Age": ages, "Probability": predict_proba(model, sweep)})
        fig = px.line(sweep_df, x="Age", y="Probability", title="If this passenger's age were different")
        fig.add_vline(x=passenger["Age"], line_dash="dot", line_color="gray")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], height=320)
        st.plotly_chart(fig, width="stretch")
    with c2:
        grid = []
        for sex in ["male", "female"]:
            for pc in [1, 2, 3]:
                alt = {**passenger, "Sex": sex, "Pclass": pc, "Fare": typical_fare(pc),
                       "Title": guess_title(sex, passenger["Age"], passenger["SibSp"])}
                grid.append({"Sex": sex.title(), "Class": CLASS_LABELS[pc],
                             "Probability": float(predict_proba(model, pd.DataFrame([alt]))[0])})
        g = pd.DataFrame(grid)
        fig = px.bar(g, x="Class", y="Probability", color="Sex", barmode="group",
                     title="Same person, different sex / class",
                     color_discrete_map={"Male": BLUE, "Female": RED})
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], height=320)
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------- tab 2: explore
def tab_explore() -> None:
    df = get_data()
    f1, f2, f3, f4 = st.columns(4)
    sexes = f1.multiselect("Sex", ["male", "female"], default=["male", "female"])
    classes = f2.multiselect("Class", list(CLASS_LABELS.values()), default=list(CLASS_LABELS.values()))
    age_lo, age_hi = f3.slider("Age range", 0, 80, (0, 80))
    ports = f4.multiselect("Embarked", ["S", "C", "Q"], default=["S", "C", "Q"])

    sel = df[df["Sex"].isin(sexes) & df["Class"].isin(classes) & df["Embarked"].isin(ports)
             & (df["Age"].isna() | df["Age"].between(age_lo, age_hi))]
    if sel.empty:
        st.warning("No passengers match these filters.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Passengers", len(sel))
    m2.metric("Survived", int(sel["Survived"].sum()))
    m3.metric("Survival rate", f"{sel['Survived'].mean():.0%}",
              delta=f"{(sel['Survived'].mean() - df['Survived'].mean()) * 100:+.0f} pts vs. all passengers")

    c1, c2 = st.columns(2)
    by_sex = sel.groupby("Sex")["Survived"].mean().reset_index()
    c1.plotly_chart(px.bar(by_sex, x="Sex", y="Survived", title="Survival rate by sex",
                           color="Sex", color_discrete_map={"male": BLUE, "female": RED})
                    .update_layout(yaxis_tickformat=".0%", showlegend=False, yaxis_title="Survival rate"),
                    width="stretch")
    by_cls = sel.groupby("Class")["Survived"].mean().reindex(CLASS_LABELS.values()).dropna().reset_index()
    c2.plotly_chart(px.bar(by_cls, x="Class", y="Survived", title="Survival rate by class")
                    .update_traces(marker_color=BLUE)
                    .update_layout(yaxis_tickformat=".0%", yaxis_title="Survival rate"),
                    width="stretch")

    c3, c4 = st.columns(2)
    c3.plotly_chart(px.histogram(sel.dropna(subset=["Age"]), x="Age", color="Outcome", nbins=30,
                                 barmode="overlay", opacity=.7, title="Age distribution",
                                 color_discrete_map={"Survived": GREEN, "Did not survive": RED}),
                    width="stretch")
    heat = sel.pivot_table(index="Sex", columns="Class", values="Survived", aggfunc="mean")
    c4.plotly_chart(px.imshow(heat, text_auto=".0%", color_continuous_scale="Blues", zmin=0, zmax=1,
                              title="Survival rate: sex × class").update_layout(coloraxis_showscale=False),
                    width="stretch")

    with st.expander("Show raw data"):
        st.dataframe(sel.drop(columns=["Class", "Outcome"]), width="stretch", hide_index=True)


# ---------------------------------------------------------------- tab 3: model lab
def tab_model(metrics: dict) -> None:
    t, base = metrics["test"], metrics["baseline_original_notebook"]
    st.markdown(f"**Selected model:** {metrics['best_model']}  ·  evaluated on "
                f"{metrics['n_test_rows']} passengers never used for training.")
    c = st.columns(4)
    c[0].metric("Accuracy", f"{t['accuracy']:.1%}", f"{(t['accuracy'] - base['accuracy']) * 100:+.1f} pts vs. original")
    c[1].metric("Precision", f"{t['precision']:.1%}")
    c[2].metric("Recall", f"{t['recall']:.1%}")
    c[3].metric("ROC-AUC", f"{t['roc_auc']:.3f}", f"{t['roc_auc'] - base['roc_auc']:+.3f} vs. original")

    st.subheader("Model comparison (5-fold cross-validation)")
    comp = pd.DataFrame(metrics["comparison"])
    st.dataframe(comp.style.format({c: "{:.3f}" for c in comp.columns if c != "Model"})
                 .highlight_max(subset=["CV Accuracy", "CV ROC-AUC"], color="#cfe8d8"),
                 width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    roc = t["roc_curve"]
    fig = go.Figure([go.Scatter(x=roc["fpr"], y=roc["tpr"], name="Model", line=dict(color=BLUE, width=3)),
                     go.Scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(dash="dash", color="gray"))])
    fig.update_layout(title=f"ROC curve (AUC = {t['roc_auc']:.3f})", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", height=380)
    c1.plotly_chart(fig, width="stretch")
    cm = np.array(t["confusion_matrix"])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Predicted: died", "Predicted: survived"], y=["Actually died", "Actually survived"],
                    title="Confusion matrix (hold-out set)")
    fig.update_layout(coloraxis_showscale=False, height=380)
    c2.plotly_chart(fig, width="stretch")

    imp = pd.DataFrame(metrics["feature_importance"]).sort_values("Importance")
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", error_x="Std",
                 title="Permutation importance: how much accuracy drops when a feature is shuffled")
    fig.update_traces(marker_color=BLUE)
    st.plotly_chart(fig, width="stretch")

    with st.expander("What changed vs. the original notebook?"):
        st.markdown(
            "- **New features:** passenger title (from the name), family size, travelling alone, cabin recorded.\n"
            "- **No data leakage:** imputing and scaling are learned from training data only, inside an sklearn Pipeline.\n"
            "- **Class treated as a category** instead of a number.\n"
            "- **Four models compared** with cross-validation instead of one model on one split.\n"
            "- **Stratified split** so survivors/non-survivors are balanced in both sets.")


# ---------------------------------------------------------------- tab 4: batch
REQUIRED = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def tab_batch(model) -> None:
    st.markdown("Upload a CSV of passengers to predict them all at once. Required columns: "
                f"`{', '.join(REQUIRED)}`. Optional: `Name`, `Cabin`, `PassengerId`.")
    up = st.file_uploader("Passenger CSV", type="csv")
    use_sample = st.button("Or try it with the Kaggle test.csv (418 passengers)")
    if use_sample:
        st.session_state["batch_df"] = load_test()
    if up is not None:
        try:
            st.session_state["batch_df"] = pd.read_csv(up)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read that file: {exc}")
            return

    df = st.session_state.get("batch_df")
    if df is None:
        return
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)}")
        return

    prob = predict_proba(model, df)
    out = df.copy()
    out["SurvivalProbability"] = prob.round(3)
    out["PredictedSurvived"] = (prob >= 0.5).astype(int)

    c1, c2, c3 = st.columns(3)
    c1.metric("Passengers", len(out))
    c2.metric("Predicted survivors", int(out["PredictedSurvived"].sum()))
    c3.metric("Predicted survival rate", f"{out['PredictedSurvived'].mean():.0%}")
    st.plotly_chart(px.histogram(out, x="SurvivalProbability", nbins=20,
                                 title="Distribution of predicted probabilities").update_traces(marker_color=BLUE),
                    width="stretch")
    st.dataframe(out, width="stretch", hide_index=True)

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button("⬇️ Download predictions (CSV)", buf.getvalue(), "titanic_predictions.csv", "text/csv")
    if "PassengerId" in out.columns:
        sub = out[["PassengerId", "PredictedSurvived"]].rename(columns={"PredictedSurvived": "Survived"})
        st.download_button("⬇️ Download Kaggle submission format", sub.to_csv(index=False),
                           "submission.csv", "text/csv")


# ---------------------------------------------------------------- main
def main() -> None:
    st.title("🚢 Titanic Survival Predictor")
    st.caption("Machine-learning demo built on the classic Kaggle Titanic dataset. "
               "Change the passenger in the sidebar and watch the prediction update.")
    model, metrics = get_model_and_metrics()
    passenger = sidebar_passenger()
    st.sidebar.divider()
    st.sidebar.caption(f"Model: {metrics['best_model']} · hold-out accuracy {metrics['test']['accuracy']:.1%}")

    t1, t2, t3, t4 = st.tabs(["🔮 Predict", "📊 Explore data", "🧪 Model lab", "📁 Batch predict"])
    with t1:
        tab_predict(model, passenger)
    with t2:
        tab_explore()
    with t3:
        tab_model(metrics)
    with t4:
        tab_batch(model)


main()
