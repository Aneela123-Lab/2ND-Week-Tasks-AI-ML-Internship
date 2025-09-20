# 2ND-Week-Tasks-AI-ML-Internship

🧠### *Task 2: End-to-End ML Pipeline for Customer Churn Prediction*
🎯 Objective

Build a reusable and production-ready machine learning pipeline for predicting customer churn using Scikit-learn's Pipeline API.

📂 Dataset

Telco Customer Churn Dataset
A publicly available dataset containing demographic, account, and usage details of telecom customers along with their churn status.

🔧 Steps Performed

Data Preprocessing with Pipelines

Handled missing values

Label encoding for binary categorical columns

One-hot encoding for multi-class categorical features

Scaled numerical features using StandardScaler

Combined all transformations into a single ColumnTransformer

Pipeline Construction

Built full ML pipeline using Scikit-learn’s Pipeline API

Included both preprocessing and model training stages

Model Training

Trained two classifiers:

LogisticRegression

RandomForestClassifier

Hyperparameter Tuning

Used GridSearchCV to find the best hyperparameters for both models

Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Model Export

Exported the final pipeline using joblib for production use

🧪 Technologies Used

Python 3.x

Scikit-learn

Pandas

Numpy

Joblib

Matplotlib / Seaborn (optional for visualizations)

📈 Results

Achieved high accuracy and F1-score on the test set using Random Forest.

Exported model can be directly used to predict churn for new customer data.

🚀 Skills Gained

End-to-end ML pipeline construction

Efficient preprocessing using Pipeline and ColumnTransformer

Hyperparameter tuning with GridSearchCV

Exporting and deploying ML models

Working with real-world tabular data

Author : 
Aneela - AI/ML Internship

Author : 
Muhammad Mustaqeem Javed - AI/ML Internship
