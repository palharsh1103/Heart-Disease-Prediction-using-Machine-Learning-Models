import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 1. Load the dataset
df = pd.read_csv("heart.csv")

# 2. Define features and target
X = df.drop("target", axis=1)
y = df["target"]

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the KNN model (best performing model in your case)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# 6. Evaluate the model (optional)
y_pred = knn_model.predict(X_test_scaled)
print("\nModel Evaluation on Test Data:")
print(classification_report(y_test, y_pred))

# 7. Take input from user
print("\n🔵 Please enter the following patient details:")

feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                 'restecg', 'thalach', 'exang', 'oldpeak',
                 'slope', 'ca', 'thal']

user_input = []
for feature in feature_names:
    value = float(input(f"{feature}: "))
    user_input.append(value)

# 8. Convert input to DataFrame with feature names
input_df = pd.DataFrame([user_input], columns=feature_names)

# 9. Scale the input
user_input_scaled = scaler.transform(input_df)

# 10. Make prediction
prediction = knn_model.predict(user_input_scaled)

# 11. Output result
print("\n🔍 Prediction Result:")
if prediction[0] == 1:
    print("🔴 The model predicts that the patient **has heart disease**.")
else:
    print("🟢 The model predicts that the patient **does NOT have heart disease**.")