# 🚢 Titanic Survival Predictor

An end-to-end machine-learning project with an **interactive web app**. Enter a passenger's details and the model estimates their chance of surviving the Titanic disaster, explains *why*, and lets you explore the data and compare models, all in the browser.

Built with **Python, scikit-learn, Streamlit and Plotly** on the classic [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic).

<!-- Add a screenshot or GIF of the app here: ![App screenshot](assets/screenshot.png) -->

## ✨ Features

| Tab | What it does |
| --- | --- |
| 🔮 **Predict** | Set age, class, sex, fare, family and port in the sidebar → live survival probability gauge, a per-factor explanation, and a *what-if* explorer (age sweep, sex × class comparison). Includes quick presets. |
| 📊 **Explore data** | Filter passengers and see survival rates, age distributions and a sex × class heatmap update instantly. |
| 🧪 **Model lab** | Cross-validated comparison of 4 models, ROC curve, confusion matrix, permutation feature importance, and a comparison with the original baseline. |
| 📁 **Batch predict** | Upload a CSV of passengers, get predictions for all of them, download the results (or a Kaggle-format `submission.csv`). |

## 🚀 Quick start

```bash
git clone https://github.com/<your-username>/titanic-survival-predictor.git
cd titanic-survival-predictor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py                  # optional: trains, compares models, saves artifacts
streamlit run app.py             # opens http://localhost:8501
```

If you skip `python train.py`, the app trains the model automatically the first time it starts.

Run the tests with `pip install -r requirements-dev.txt && pytest`.

## 📈 Results

Evaluated on a stratified 20 % hold-out set (179 passengers) that was never used for training or model selection:

| | Original notebook | This project |
| --- | --- | --- |
| Model | Logistic Regression, 7 raw features | Logistic Regression, engineered features, Pipeline |
| Accuracy | 80.4 % | **83.2 %** |
| ROC-AUC | 0.842 | **0.872** |
| Precision / Recall (survived) | n/a | 80.0 % / 75.4 % |

Model selection used 5-fold cross-validation on the training split only (Logistic Regression, Random Forest, Gradient Boosting and SVM all landed within about 2 points of each other, around 81-83 %). With 891 rows, differences that small are mostly noise, so the simplest model that does as well as the rest was kept.

The most important features by permutation importance are **Title** (extracted from the passenger's name), **Sex**, **Age** and **Class**.

## 🔧 What was improved over the original notebook

- **Feature engineering**: passenger title, family size, travelling alone, cabin recorded.
- **No data leakage**: the original filled missing ages using the mean of the *whole* dataset before splitting. Imputation and scaling are now learned from training data only, inside a scikit-learn `Pipeline`.
- **Passenger class is categorical** (1st / 2nd / 3rd), not a number that implies 3rd is "three times" 1st.
- **Honest evaluation**: stratified split, cross-validation, ROC-AUC, precision/recall and a confusion matrix instead of accuracy alone.
- **Reusable code** in a `titanic/` package with unit tests and CI, instead of one notebook.
- **Explainability**: every prediction comes with a breakdown of what raised or lowered the probability.

## 🗂️ Project structure

```
├── app.py                 # Streamlit app (4 tabs)
├── train.py               # Train, compare, save model + metrics + submission
├── titanic/
│   ├── data.py            # Loading + stateless feature engineering
│   └── model.py           # Pipelines, CV comparison, evaluation, explanations
├── tests/                 # pytest suite
├── data/                  # Kaggle train.csv / test.csv / gender_submission.csv
├── models/                # metrics.json, submission.csv (model.joblib is rebuilt locally)
├── notebooks/             # Original exploration notebook
├── docs/                  # Original project report and testing notes
└── .github/workflows/     # CI: runs the tests on every push
```

## ⚠️ Limitations

- Trained on only 891 labelled passengers, so predictions are rough statistical estimates and not precise.
- The model reflects the social norms of 1912 ("women and children first", class privilege). It describes what happened in this dataset; it says nothing about any real individual.
- Cabin, ticket and family-group information are only partly used.

## 🛣️ Ideas for future work

- Hyper-parameter tuning (e.g. `GridSearchCV` / Optuna) and a stacked ensemble.
- Extract deck from cabin numbers and group travellers by ticket.
- SHAP values for richer explanations.
- Deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud) (point it at `app.py`).

## 📄 License & credits

MIT License (see `LICENSE`). Dataset: [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic).
Created by **YOUR NAME** as a university AI course project at Sukkur IBA University.
