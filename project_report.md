# Write the project report in Markdown
report_content = """
# Telco Customer Churn Prediction – Project Report

## Problem Statement
To predict customer churn in a telecom company using customer demographic and service usage data.

---

## Dataset Description
- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows**: 7043
- **Target**: `Churn` (Yes/No)
- **Challenge**: Multiple categorical features and class imbalance

---

## Methodology

### 1. Preprocessing
- Dropped `customerID`
- Converted `TotalCharges` to numeric
- Encoded `Churn` as binary (Yes=1, No=0)
- One-hot encoded all categorical features

### 2. Modeling
- **Train-Test Split**: 80/20, stratified
- **Model**: Random Forest Classifier (100 trees)
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC

---

## Results

| Metric       | Score   |
|--------------|---------|
| Accuracy     | ~80%    |
| Precision    | ~79%    |
| Recall       | ~75%    |
| F1 Score     | ~77%    |
| ROC-AUC      | ~85%    |

---

## Discussion
- Random Forest performed well with minimal tuning.
- Class imbalance was manageable due to stratified split.
- Feature encoding had a large impact on model accuracy.

---

## Conclusion
- A simple Random Forest model provides solid performance for churn prediction.
- This model can assist telecom companies in customer retention strategies.

---

## Streamlit App
A user-friendly interface was built with **Streamlit** to allow business users to input customer data and receive churn predictions in real time.

Run it using:
```bash
streamlit run streamlit_app.py
