Predicting Employee Attrition Using Data Mining Techniques

This project analyzes the IBM HR Analytics Employee Attrition & Performance dataset to identify key factors contributing to employee turnover and to build machine learning models that predict whether an employee is likely to leave the company. The project implements Decision Tree and Random Forest classifiers, applies SMOTE to handle class imbalance, and evaluates model performance using accuracy, precision, recall, and F1-score.

Dataset

IBM HR Analytics Attrition Dataset
Kaggle link: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

The dataset contains 1,470 employees with 35 features, including:
Demographics (Age, Gender, Education, etc.)
Work environment (JobRole, Department, WorkLifeBalance)
Satisfaction metrics (JobSatisfaction, EnvironmentSatisfaction)
Compensation (MonthlyIncome, PercentSalaryHike)
Target variable: Attrition (Yes/No)

Technologies Used
Python 3
pandas, numpy
scikit-learn
imbalanced-learn (SMOTE)
matplotlib, seaborn

Installation
1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

2. Install dependencies
pip install -r requirements.txt
