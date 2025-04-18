import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Ensure 'model' directory exists
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("diabetes.csv")

# Replace 0s with NaN for specific columns
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# Fill NaN with median grouped by Outcome
def median_target(var):
    temp = df[df[var].notnull()]
    return temp[[var, 'Outcome']].groupby(['Outcome']).median().reset_index()

for col in cols_with_zero:
    medians = median_target(col)
    df.loc[(df['Outcome'] == 0) & (df[col].isnull()), col] = medians.loc[0, col]
    df.loc[(df['Outcome'] == 1) & (df[col].isnull()), col] = medians.loc[1, col]

# Cap outliers in Insulin column using IQR method
Q1 = df["Insulin"].quantile(0.25)
Q3 = df["Insulin"].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
df.loc[df["Insulin"] > upper, "Insulin"] = upper

# Remove global outliers using Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=10)
outlier_scores = lof.fit_predict(df)
df = df[outlier_scores != -1]

# Feature Engineering
df["NewBMI"] = pd.cut(df["BMI"],
                      bins=[0,18.5,24.9,29.9,34.9,39.9,100],
                      labels=["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"])

df["NewInsulinScore"] = df["Insulin"].apply(lambda x: "Normal" if 16 <= x <= 166 else "Abnormal")

df["NewGlucose"] = pd.cut(df["Glucose"],
                          bins=[0,70,99,126,500],
                          labels=["Low", "Normal", "Overweight", "Secret"])

# One-hot encoding
df = pd.get_dummies(df, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)

# Define features and target
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

# Separate for scaling
categorical_cols = [col for col in X.columns if any(prefix in col for prefix in ["NewBMI", "NewInsulinScore", "NewGlucose"])]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Scale numerical features
scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X[numerical_cols]), columns=numerical_cols, index=X.index)

# Combine with categorical features
X_final = pd.concat([X_scaled, X[categorical_cols]], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train XGBoost model
from xgboost import XGBClassifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Save the model and scaler to 'model/' folder
with open("model/xgb_diabetes_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save column order (needed for input formatting later)
with open("model/input_columns.pkl", "wb") as f:
    pickle.dump(X_final.columns.tolist(), f)
