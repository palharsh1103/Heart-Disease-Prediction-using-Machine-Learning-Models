import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the trained model and scaler (if saved)
# For now, we'll train the model in this script for simplicity

# Load dataset
df = pd.read_csv("heart.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split & scale
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train best model (KNN in this case)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# UI Design
st.title("💓 Heart Disease Prediction App")
st.write("Enter patient details below to predict the risk of heart disease.")

# Sidebar inputs
def user_input_features():
    age = st.slider("Age", 20, 90, 45)
    sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure (trestbps)", 90, 200, 120)
    chol = st.slider("Serum Cholestoral (chol)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved (thalach)", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.slider("ST depression induced (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Predict
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)

# Display result
st.subheader("🔍 Prediction Result")
if prediction[0] == 1:
    st.error("🔴 The model predicts that the patient **has heart disease**.")
else:
    st.success("🟢 The model predicts that the patient **does NOT have heart disease**.")