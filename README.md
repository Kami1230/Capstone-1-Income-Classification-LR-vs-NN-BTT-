# Breakthrough Tech ML Foundations Capstone
## Income Classification: Logistic Regression vs. Neural Network

A binary classification project predicting whether an individual's annual income exceeds $50K, using 1994 U.S. Census data. Built as part of Break Through Tech's Machine Learning Foundations coursework, then cleaned up as a standalone project.

## Problem

Predict `income > $50K` vs. `income <= $50K` from demographic and employment features (age, workclass, marital status, occupation, hours worked, etc.). Framed around a business use case: this kind of model could feed into decisions like loan or credit eligibility screening — which shapes how the project weighs interpretability against raw performance.

## Approach

- **EDA**: Identified a ~75/25 class imbalance, skewed numerical features (`capital-gain`, `capital-loss`), and missing values in `workclass`, `occupation`, and `native-country`.
- **Ethical review**: Flagged `race` and `sex_selfID` as explicit demographic attributes, and features like `marital-status`, `education`, and `occupation` as potential structural proxies for the same. `race` was dropped from training for this reason.
- **Data prep**: Dropped low-value/redundant columns, imputed missing values (mean for numeric, placeholder for categorical), one-hot encoded categorical features, and binarized the label.
- **Model 1 — Logistic Regression**: Tuned via `GridSearchCV` over regularization strength.
- **Model 2 — Neural Network**: A small feedforward network (2 hidden layers, ReLU, sigmoid output), trained with SGD and manually tuned epochs/learning rate.
- **Comparison**: Evaluated both on accuracy and F1 (F1 weighted more heavily given the class imbalance).

## Results

| Metric   | Logistic Regression | Neural Network |
|----------|---------------------|----------------|
| Accuracy | 85.45%              | 86.31%         |
| F1 Score | 0.6713              | 0.6853         |

## Key Finding

The neural network edges out logistic regression on both metrics, but the gain is small relative to what it costs: longer training time, more hyperparameters to tune, and a much less interpretable model. **Recommendation: logistic regression** — in a context like loan/credit eligibility, being able to explain *why* a prediction was made is often a practical and regulatory requirement, not a nice-to-have, and LR is far cheaper to retrain and audit for fairness issues over time.

## Tech Stack

Python, pandas, NumPy, scikit-learn (LogisticRegression, GridSearchCV), TensorFlow/Keras, seaborn, matplotlib

## Repo Contents

- `income_classification.ipynb` - full notebook: EDA, data preparation, both models, and analysis
- `censusData.csv` - dataset used for training and testing the model.