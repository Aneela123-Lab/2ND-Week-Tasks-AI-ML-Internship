# 2ND-Week-Tasks-AI-ML-Internship
# Task 2: End-to-End ML Pipeline for Customer Churn Prediction
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
Aneela - AI/ML Internship

# task 1 model (BERT) 
Objective

To fine-tune a transformer-based model (BERT) to classify news headlines into topic categories using the AG News Dataset.

📂 Dataset

AG News Dataset from Hugging Face Datasets

Contains 4 news categories:

World

Sports

Business

Sci/Tech

Each sample includes a:

Title (used as input text)

Label (used as class target)

🔧 Steps Performed

Dataset Loading and Preprocessing

Loaded dataset using load_dataset("ag_news")

Tokenized using BertTokenizerFast from bert-base-uncased

Truncated and padded input sequences

Model Initialization

Used BertForSequenceClassification with 4 output labels

Configured training using TrainingArguments and Trainer

Model Training

Fine-tuned on training set

Evaluated on test set using:

Accuracy

F1-score

Deployment

Built an interactive demo using Gradio

Users can input a headline and get predicted topic in real-time

📈 Evaluation Metrics
Metric	Value (Example)
Accuracy	✅ Implemented
F1-Score	✅ Implemented

Note: Metrics were calculated using the Hugging Face Trainer evaluation API.

🧪 Technologies Used

Python 3.x

Hugging Face Transformers

Datasets (AG News)

PyTorch

Gradio (for UI)

scikit-learn (for evaluation)

 # Key Features
Feature	Implemented
BERT fine-tuning on AG News	✅
Text tokenization and batching	✅
Model evaluation (F1 & Accuracy)	✅
Gradio deployment for demo	✅
🚀 Skills Gained

Working with transformer-based NLP models

Text classification using pre-trained models

Evaluation using industry-standard metrics

Lightweight UI deployment with Gradio
# Author : 
# Aneela - AI/ML Intern

End-to-end ML/NLP pipeline design


# Multimodal House Price Prediction
This project demonstrates how to build a multimodal deep learning model using PyTorch to predict house prices. The model leverages two types of data: tabular features (e.g., bedrooms, bathrooms, square footage) and simulated image features, combining them to make a final prediction.

# Features
Data Generation: Automatically generates a sample housing_data.csv file if one doesn't exist, which includes tabular data and dummy image paths.
Simulated Image Features: Uses a SimulatedCNNFeatureExtractor to create random feature vectors, standing in for features that would be extracted by a real Convolutional Neural Network (CNN) from house images.
Data Preprocessing: Standardizes tabular features using StandardScaler from scikit-learn.
Multimodal Architecture: A simple feed-forward neural network is defined in PyTorch (MultimodalHousePriceModel) to process the concatenated tabular and image features.
Training & Evaluation: The model is trained using Adam optimizer and MSELoss. It is then evaluated on a validation set using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to measure performance.
Sample Prediction: Provides a sample prediction to demonstrate the model's output and calculate the error against the actual price.
Requirements
The code is written in Python and requires the following libraries:

pandas
numpy
torch
scikit-learn
Code Structure
create_sample_csv(): Function to generate a synthetic dataset for demonstration purposes.
SimulatedCNNFeatureExtractor: A class that simulates a CNN's role in extracting features from images.
HousingDataset: A custom PyTorch Dataset class to handle the combined features and targets.
MultimodalHousePriceModel: The PyTorch nn.Module defining the neural network architecture.
Training Loop
The code block for training the model over a set number of epochs.

# Evaluation
A section to calculate and display key performance metrics (MAE, RMSE) on a validation set.



