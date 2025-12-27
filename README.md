# Loan Approval Prediction

A machine learning web application that predicts whether a loan will be approved or not based on user inputs such as income, credit history, loan amount, employment status, and more.

---

## Project Overview

This project is built to assist financial institutions in automating the process of loan approval using historical data and predictive modeling. The model is trained using several classification algorithms, and a Streamlit web interface allows users to interact with the model easily.

---

## Exploratory Data Analysis (EDA)

Before model building, detailed EDA was performed:

* Handled missing values using mode/mean imputation.
* Removed outliers from `ApplicantIncome` and `CoapplicantIncome` using IQR.
* Found `Credit_History` to be highly correlated with `Loan_Status`.
* Detected and resolved class imbalance using SMOTE.

---

## Models Tried

Several classification models were trained and evaluated:

* Logistic Regression
* Support Vector Classifier (SVC)
* Decision Tree
* Random Forest (Finalized)
* XGBoost

 **Random Forest** was chosen for its superior performance on test data with balanced bias-variance tradeoff.

---

## Tech Stack

* Python
* Pandas, NumPy, Matplotlib, Seaborn
* Scikit-learn, XGBoost
* imbalanced-learn (SMOTE)
* Streamlit

---

## Project Structure

```
Loan_Approval_Prediction/
├── loan_prediction.csv               # Dataset
├── loan_approval_app.py              # Streamlit Web App
├── model_training_script.py          # ML Training Script
├── loan_approval_model.pkl           # Trained Random Forest Model
├── scaler.pkl                        # MinMaxScaler for feature scaling
├── feature_names.pkl                 # Feature list used during training
├── model_metrics.pkl                 # Metrics for model evaluation
├── requirements.txt                  # Python dependencies
├── README.md                         # Project Documentation
└── loan_approval_prediction.ipynb    # Jupyter Notebook version of the pipeline
```

---

## Data Preprocessing & Training

* Missing values filled using mode or mean
* Outliers removed using IQR
* Categorical variables converted with One-Hot Encoding
* Features scaled using MinMaxScaler
* SMOTE applied to resolve class imbalance
* Train-test split: 80-20 ratio
* Trained using RandomForestClassifier (100 estimators, default params)

---

## Deployment

### Setup

Clone the repository:

```bash
git clone https://github.com/bhanuvi17/Loan_approval_prediction.git
cd Loan_Approval_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run loan_approval_app.py
```

---

## Model Performance

Model trained using Random Forest shows solid performance on test set.

* Accuracy: \~**85-87%** (varies slightly depending on test split)
* Balanced results after handling data imbalance using SMOTE

---

## GitHub Repository

[Loan Approval Prediction on GitHub](https://github.com/bhanuvi17/Loan_approval_prediction.git)

---

## Screenshots

🔹 **Home Page**  
![Home Page](https://github.com/bhanuvi17/Loan_approval_prediction/blob/1767495cd3b61668a34ae84b3605de2f1cee7fdc/Screenshot%202025-05-21%20203805.png)

---

## Author

**M Bhanuprakash**
B.E. in Computer Science | Aspiring ML Engineer
 [bhanuprakash1722004@gmail.com](mailto:bhanuprakash1722004@gmail.com)

---

## 💡 Future Work

* Add live deployment on Streamlit Cloud / Hugging Face Spaces
* Integrate SHAP for model interpretability
* Enable batch upload for bulk predictions
