# 🫀 Heart Disease Prediction using Machine Learning

This project leverages multiple machine learning models to predict the likelihood of heart disease in individuals using clinical and demographic data. The aim is to support early diagnosis and improve the accuracy of predictions using well-tuned ML techniques and user-friendly deployment.

---

## 📌 Features

- Data cleaning and preprocessing
- Multiple machine learning algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - XGBoost
  - Neural Network (MLPClassifier)
- Performance metrics: Accuracy, Precision, Recall, F1-Score
- Visual analysis with graphs and comparison charts
- User-friendly Streamlit app for predictions
- Human-readable label mappings (e.g., `Sex: Male/Female` instead of `0/1`)

---

## 📊 Dataset

- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Attributes Include:**
  - Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Max Heart Rate, etc.
  - **Target variable:** Presence (`1`) or Absence (`0`) of heart disease

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Jupyter Notebook / Google Colab
- VS Code

---

## 🧪 Project Files

| File | Purpose |
|------|---------|
| `Heart Disease Model Evaluation.ipynb` | Colab/Jupyter Notebook for training, evaluating and comparing ML models to find the most effective one (KNN performs best) |
| `heart_app.py` | Streamlit web application for real-time user input and prediction |
| `evaluate_model_using_sample_input.py` | Command-line interface to test predictions directly in the terminal using predefined or manual input |

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/palharsh1103/Heart-Disease-Prediction-using-Machine-Learning-Models.git
cd Heart-Disease-Prediction-using-Machine-Learning-Models
