# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import roc_curve, auc

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("❤️ Heart Disease Prediction App")

# -------------------------------
# Load the trained model & results
# -------------------------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model_results.pkl", "rb") as f:
        results = pickle.load(f)

except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

# -------------------------------
# Input form
# -------------------------------
st.sidebar.header("Input Parameters")
def user_input_features():
    Age = st.sidebar.number_input("Age", 1, 120, 50)
    Sex = st.sidebar.selectbox("Sex", ["M", "F"])
    ChestPainType = st.sidebar.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
    RestingBP = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
    Cholesterol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
    FastingBS = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    RestingECG = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    MaxHR = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
    ExerciseAngina = st.sidebar.selectbox("Exercise Angina", ["Y", "N"])
    Oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
    ST_Slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

    data = {
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("User Input Parameters")
st.dataframe(input_df)

# -------------------------------
# Prediction
# -------------------------------
try:
    prediction = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]  # probability of heart disease

    st.subheader("Prediction")
    heart_status = "❤️ Heart Disease" if prediction == 1 else "💚 No Heart Disease"
    st.write(f"Predicted Status: **{heart_status}**")
    st.write(f"Probability of Heart Disease: **{pred_proba:.2f}**")

except Exception as e:
    st.error(f"Error in prediction: {e}")
    st.stop()

# -------------------------------
# Graphical Representation
# -------------------------------
st.subheader("📊 Prediction Probability Visualization")

# Bar chart
prob_df = pd.DataFrame({
    "Outcome": ["No Heart Disease", "Heart Disease"],
    "Probability": [1 - pred_proba, pred_proba]
})
st.bar_chart(prob_df.set_index("Outcome"))

# Pie chart
fig1, ax1 = plt.subplots()
ax1.pie([1 - pred_proba, pred_proba],
        labels=["No Heart Disease", "Heart Disease"],
        autopct="%1.1f%%",
        colors=["#4CAF50", "#F44336"],
        startangle=90)
ax1.axis("equal")
st.pyplot(fig1)

# -------------------------------
# Feature Importance Plot
# -------------------------------
if "feature_importance" in results:
    st.subheader("🧠 Feature Importance")
    importances = results["feature_importance"]
    feat_df = pd.DataFrame({
        "Feature": list(importances.keys()),
        "Importance": list(importances.values())
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis", ax=ax2)
    st.pyplot(fig2)
else:
    st.warning("Feature importance data not found.")

# -------------------------------
# ROC Curve Plot
# -------------------------------
if "roc_curve" in results:
    st.subheader("📈 ROC Curve")
    fpr, tpr, roc_auc = results["roc_curve"]
    fig3, ax3 = plt.subplots()
    ax3.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
    ax3.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("Receiver Operating Characteristic (ROC)")
    ax3.legend(loc="lower right")
    st.pyplot(fig3)
else:
    st.warning("ROC curve data not found.")

# -------------------------------
# Model Comparison Plot
# -------------------------------
if "model_accuracy" in results:
    st.subheader("🤖 Model Comparison")
    model_acc_df = pd.DataFrame(list(results["model_accuracy"].items()), columns=["Model", "Accuracy"])
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Accuracy", y="Model", data=model_acc_df, palette="magma", ax=ax4)
    st.pyplot(fig4)
else:
    st.warning("Model comparison data not found.")
