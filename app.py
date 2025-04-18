from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and expected input columns
model = pickle.load(open('model/xgb_diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))
input_columns = pickle.load(open('model/input_columns.pkl', 'rb'))

# Base input features from user (before one-hot encoding)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        input_data = [float(request.form[feature]) for feature in features]
        input_dict = dict(zip(features, input_data))
        df = pd.DataFrame([input_dict])

        # Feature Engineering
        df["NewBMI"] = pd.cut(df["BMI"],
                              bins=[0,18.5,24.9,29.9,34.9,39.9,100],
                              labels=["Underweight","Normal","Overweight","Obesity 1","Obesity 2","Obesity 3"])

        df["NewInsulinScore"] = df["Insulin"].apply(lambda x: "Normal" if 16 <= x <= 166 else "Abnormal")

        df["NewGlucose"] = pd.cut(df["Glucose"],
                                  bins=[0,70,99,126,500],
                                  labels=["Low","Normal","Overweight","Secret"])

        # One-hot encode new categorical features
        df = pd.get_dummies(df, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)

        # Scale numerical features
        categorical_cols = [col for col in df.columns if any(prefix in col for prefix in ["NewBMI", "NewInsulinScore", "NewGlucose"])]
        numerical_cols = [col for col in df.columns if col not in categorical_cols]
        df[numerical_cols] = scaler.transform(df[numerical_cols])

        # Ensure all expected columns are present
        for col in input_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[input_columns]

        # Make prediction
        prediction = model.predict(df)[0]
        result = 'Diabetic' if prediction == 1 else 'Not Diabetic'
    except Exception as e:
        result = f"Error: {e}"
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
