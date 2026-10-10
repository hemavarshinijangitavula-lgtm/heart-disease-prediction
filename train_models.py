import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("heart.csv")

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

categorical_cols = [
    "Sex", "ChestPainType", "RestingECG",
    "ExerciseAngina", "ST_Slope"
]

numerical_cols = [
    "Age", "RestingBP", "Cholesterol",
    "FastingBS", "MaxHR", "Oldpeak"
]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = {}
pipelines = {}

# -----------------------------
# Train & Evaluate
# -----------------------------
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    results[name] = {"accuracy": acc, "roc_auc": roc}
    pipelines[name] = pipeline

# -----------------------------
# Save Best Model (Random Forest)
# -----------------------------
best_model = pipelines["Random Forest"]

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("✅ Models trained")
print(results)
