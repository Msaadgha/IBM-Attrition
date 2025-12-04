import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
import joblib



CSV_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

if os.path.exists(CSV_FILENAME):
    print(f"✓ Found dataset: {CSV_FILENAME}")
    df = pd.read_csv(CSV_FILENAME)
else:
    raise FileNotFoundError(
        f"ERROR: Cannot find the dataset.\n"
        f"Place '{CSV_FILENAME}' in the same folder as this script."
    )

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset shape:", df.shape)


print("\nMissing values per column:")
print(df.isnull().sum())

print("\nAttrition value counts:")
print(df["Attrition"].value_counts())

plt.figure(figsize=(8,4))
sns.countplot(data=df, x='JobSatisfaction', hue='Attrition')
plt.title("Attrition by Job Satisfaction")
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome')
plt.title("Monthly Income vs Attrition")
plt.show()



# Remove ID-like column
if "EmployeeNumber" in df.columns:
    df = df.drop(columns=["EmployeeNumber"])

# Encode target
df["Attrition_Flag"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Feature matrix and target
X = df.drop(columns=["Attrition", "Attrition_Flag"])
y = df["Attrition_Flag"]

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric Columns:", num_cols)
print("Categorical Columns:", cat_cols)

# Build transformers
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining size:", X_train.shape, "Testing size:", X_test.shape)

# Fit preprocessing
X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_pre, y_train)

print("\nAfter SMOTE balancing:")
print(pd.Series(y_train_res).value_counts())



dt_clf = DecisionTreeClassifier(random_state=42)
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)

dt_clf.fit(X_train_res, y_train_res)
rf_clf.fit(X_train_res, y_train_res)

models = {
    "Decision Tree": dt_clf,
    "Random Forest": rf_clf
}


def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n===== {name} =====")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print(f"ROC AUC:    {auc:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.show()



for name, model in models.items():
    evaluate_model(model, X_test_pre, y_test, name)


ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_features = ohe.get_feature_names_out(cat_cols)

feature_names = list(num_cols) + list(cat_features)

importances = rf_clf.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\nTop 20 Important Features:")
print(feat_imp.head(20))

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.head(15).values, y=feat_imp.head(15).index)
plt.title("Top 15 Important Features - Random Forest")
plt.xlabel("Importance")
plt.show()



joblib.dump(rf_clf, "rf_attrition_model.joblib")
joblib.dump(preprocessor, "preprocessor.joblib")

print("\n✓ Model and preprocessor saved successfully!")



def predict_single(employee_dict):
    """Predicts attrition for a single employee record."""
    df_single = pd.DataFrame([employee_dict])
    Xp = preprocessor.transform(df_single)
    proba = rf_clf.predict_proba(Xp)[0][1]
    pred = rf_clf.predict(Xp)[0]
    return {
        "Prediction": "Yes" if pred == 1 else "No",
        "Attrition_Probability": round(float(proba), 4)
    }

# Example INPUT
example_employee = {
    'Age': 35,
    'BusinessTravel': 'Travel_Rarely',
    'DailyRate': 1100,
    'Department': 'Research & Development',
    'DistanceFromHome': 10,
    'Education': 3,
    'EducationField': 'Life Sciences',
    'EnvironmentSatisfaction': 3,
    'Gender': 'Female',
    'HourlyRate': 65,
    'JobInvolvement': 3,
    'JobLevel': 2,
    'JobRole': 'Research Scientist',
    'JobSatisfaction': 3,
    'MaritalStatus': 'Single',
    'MonthlyIncome': 6000,
    'MonthlyRate': 10000,
    'NumCompaniesWorked': 1,
    'OverTime': 'No',
    'PercentSalaryHike': 12,
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 2,
    'StockOptionLevel': 0,
    'TotalWorkingYears': 8,
    'TrainingTimesLastYear': 3,
    'WorkLifeBalance': 2,
    'YearsAtCompany': 4,
    'YearsInCurrentRole': 3,
    'YearsSinceLastPromotion': 1,
    'YearsWithCurrManager': 3
}

print("\nSingle Employee Prediction Example:")
print(predict_single(example_employee))
