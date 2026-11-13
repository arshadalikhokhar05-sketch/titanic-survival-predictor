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
