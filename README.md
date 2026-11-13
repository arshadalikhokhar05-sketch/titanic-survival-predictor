
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
## 📸 Project Screenshots & video 


###  Project Demonstration Video

Watch the complete project demonstration:

**[▶ Watch Titanic Survival Prediction Demo](YOUR_VIDEO_LINK_HERE)**

The demonstration covers the application interface, passenger information input, and machine learning predictions.
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
=======
# 🚢 Titanic Survival Prediction using Machine Learning

A Machine Learning project that predicts whether a passenger would have survived the Titanic disaster using **Logistic Regression**. The project includes data preprocessing, exploratory data analysis, model training, evaluation, and an interactive web interface built with Streamlit.

## 📌 Project Overview

The Titanic Survival Prediction project uses historical passenger data to predict survival outcomes.

The model learns patterns from passenger information such as age, gender, passenger class, family members onboard, ticket fare, and boarding port.

The application allows users to enter passenger details through a graphical interface and receive a survival prediction.

## 🎯 Project Objectives

- Understand and explore a real-world dataset.
- Perform data cleaning and preprocessing.
- Handle missing values.
- Convert categorical data into numerical values.
- Visualize data using Matplotlib and Seaborn.
- Train a Logistic Regression classification model.
- Evaluate model performance using accuracy score.
- Build an interactive GUI using Streamlit.
- Predict survival for new passenger information.

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations |
| Pandas | Data manipulation and preprocessing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and evaluation |
| Logistic Regression | Classification algorithm |
| Streamlit | Graphical user interface |
| Joblib | Saving and loading the trained model |
| Jupyter Notebook | Model development and experimentation |

## 📂 Project Structure

```text
titanic-survival-prediction/
│
├── app.py
├── Titanic_prediction.ipynb
├── titanic_model.pkl
├── train.csv
├── test.csv
├── gender_submission.csv
├── requirements.txt
├── README.md
├── .gitignore
└── project_report.docx
```

## 📊 Dataset Description

The project uses the Titanic dataset from Kaggle.

Dataset source:

https://www.kaggle.com/competitions/titanic/data

Important dataset features:

| Feature | Description |
|---|---|
| PassengerId | Unique passenger identification |
| Survived | Target variable (0 = No, 1 = Yes) |
| Pclass | Passenger ticket class |
| Sex | Passenger gender |
| Age | Passenger age |
| SibSp | Siblings and spouses onboard |
| Parch | Parents and children onboard |
| Fare | Passenger ticket price |
| Embarked | Port of boarding |

## ⚙️ Data Preprocessing

The following preprocessing operations were performed:

1. Loaded the dataset using Pandas.
2. Inspected the dataset and identified missing values.
3. Filled missing Age values using the mean.
4. Filled missing Embarked values using the mode.
5. Handled missing Fare values using the median.
6. Removed unnecessary columns such as PassengerId, Name, Ticket, and Cabin.
7. Converted categorical features into numerical values.

### Categorical Encoding

**Sex:**
- Male = 0
- Female = 1

**Embarked:**
- Southampton (S) = 0
- Cherbourg (C) = 1
- Queenstown (Q) = 2

## 📈 Exploratory Data Analysis

Data visualization was performed using Matplotlib and Seaborn.

The analysis explored:

- Overall passenger survival distribution.
- Survival rates based on gender.
- Survival rates across passenger classes.
- Relationships between passenger age, fare, and survival.

## 🤖 Machine Learning Model

**Algorithm:** Logistic Regression

Logistic Regression was selected because this project is a binary classification problem.

The target variable has two possible outcomes:

- 0 = Passenger did not survive.
- 1 = Passenger survived.

### Model Training

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

The model was trained using Scikit-learn:

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

## 📊 Model Evaluation

The model was evaluated using the Accuracy Score metric.

**Testing Accuracy: Approximately 78.21%**

The model was also evaluated on training data to compare its performance on training and unseen testing samples.

## 🖥️ Graphical User Interface

An interactive GUI was developed using Streamlit.

Users can enter:

- Passenger Class
- Gender
- Age
- Number of Siblings / Spouses
- Number of Parents / Children
- Ticket Fare
- Boarding Port

After entering the information, users click the **Predict Survival** button to receive the model's prediction.

The application also displays the estimated survival probability.

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/titanic-survival-prediction.git
```

### 2. Open the project directory

```bash
cd titanic-survival-prediction
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

### 5. Open the application

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open it in your browser.

## 📌 Example Prediction

Example passenger information:

```python
[3, 0, 22, 1, 0, 7.25, 0]
```

This represents:

- Third-class passenger
- Male
- Age: 22
- One sibling or spouse
- No parents or children
- Fare: 7.25
- Embarked: Southampton

The trained model predicts one of two outcomes:

- 0 = Did not survive
- 1 = Survived

## 🔮 Future Improvements

- Experiment with Random Forest and Decision Tree algorithms.
- Improve performance through feature engineering.
- Add more interactive visualizations.
- Improve the GUI design.
- Deploy the application online.
- Compare multiple machine learning algorithms.

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Python programming for machine learning.
- Working with real-world CSV datasets.
- Data cleaning and preprocessing.
- Exploratory Data Analysis.
- Logistic Regression.
- Model training and testing.
- Model evaluation.
- Building interactive applications with Streamlit.
- Saving and loading trained machine learning models.

## 👨‍💻 Author

**Arshad Ali**

BS Computer Science Student  
Sukkur IBA University

GitHub: https://github.com/arshadalikhokhar05-sketch

---

⭐ If you find this project useful, feel free to explore the repository and share your feedback.

**Built as a practical Machine Learning project to strengthen my understanding of data science and predictive modeling.**
>>>>>>> e10d9b6a19cbad457e450bc6321f95cf6fb55a91
