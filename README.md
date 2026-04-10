# COMP263 Deep Learning – Group #3  

## Fraud Detection using Deep Learning

### Project Overview
This project focuses on detecting fraudulent credit card transactions using deep learning. The dataset is highly imbalanced, so our goal is to catch as many fraud cases as possible while improving overall model performance.

### Dataset
Download the dataset from Kaggle:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

After downloading, extract the CSV file to: `/backend/data`

### Approach
Preprocess and scale the data
Train an autoencoder on normal transactions
Use reconstruction error to detect fraud
Add a neural network classifier to improve results
