
# Telco Customer Churn Prediction – Project Report

## 1. Executive Summary

This report outlines a machine learning project designed to predict customer churn for a telecommunications company. Churn prediction is a critical challenge in industries with recurring revenue models, and this project addresses it by leveraging real-world customer data to build a predictive model. The model has been deployed through a web interface using Streamlit, allowing real-time interaction and evaluation.

---

## 2. Business Context and Objective

Telecom companies experience frequent customer attrition. Reducing churn by even a small percentage can have a significant impact on revenue. The aim of this project is to identify customers who are likely to leave (churn) based on their usage patterns, demographics, and service attributes. This allows businesses to implement retention strategies more effectively.

---

## 3. Dataset Overview

The dataset used is sourced from a real telco company and contains 7,043 records, each representing a customer. The target variable is `Churn`, indicating whether a customer has discontinued service. Key features include:

- **Demographics**: Gender, SeniorCitizen, Partner, Dependents
- **Account Info**: Tenure, MonthlyCharges, TotalCharges
- **Service Details**: InternetService, Contract Type, Tech Support, etc.

Data quality was high, with a few missing or inconsistent values in the `TotalCharges` column, which were handled appropriately during preprocessing.

---

## 4. Methodology

### Data Preprocessing

- Converted `TotalCharges` to numeric
- Removed entries with missing `TotalCharges`
- Categorical variables were encoded using one-hot encoding
- Target variable `Churn` was encoded to binary (Yes=1, No=0)

### Model Selection

A Random Forest classifier was selected due to its robustness and ability to handle both categorical and numerical variables with minimal scaling. It also provides strong baseline performance without heavy hyperparameter tuning.

---

## 5. Model Evaluation

After splitting the data into training and testing sets (80/20), the model achieved the following performance:

- **Accuracy**: ~80%
- **Precision**: ~79%
- **Recall**: ~75%
- **F1 Score**: ~77%
- **ROC-AUC Score**: ~85%

These results indicate a balanced performance across metrics, with the model being particularly effective at identifying true churn cases.

---

## 6. Deployment and Usage

The trained model was integrated into a web application using Streamlit. The app allows users to enter customer information through a form and receive real-time churn predictions. Behind the scenes, the inputs are transformed and aligned to match the model’s expected feature format before making predictions.

---

## 7. Insights and Discussion

From the model and data exploration, some key insights emerged:

- Customers on month-to-month contracts are significantly more likely to churn.
- Higher monthly charges and shorter tenure are strong indicators of churn.
- Lack of add-on services such as online security or tech support correlates with higher churn.

These insights can directly inform customer retention campaigns.

---

## 8. Conclusion and Future Work

This project successfully demonstrates how machine learning can be applied to solve a real-world business problem. The integration with Streamlit enhances accessibility and interactivity for end users. Future enhancements could include:

- Incorporating SHAP for model explainability
- Implementing cost-sensitive learning to address business costs of false positives
- Expanding to multi-class churn risk levels

Overall, the project offers a strong foundation for data-driven customer management in the telecom industry.
