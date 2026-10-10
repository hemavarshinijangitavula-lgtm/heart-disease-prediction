# generate_model_results.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score

# Load dataset
df = pd.read_csv("heart.csv")

# Features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Identify categorical and numerical columns
categorical_cols = ["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(), categorical_cols)
    ]
)

# Random Forest pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:,1]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Feature importance (only for RandomForest)
model = pipeline.named_steps["classifier"]
# Get transformed feature names
encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
all_features = numerical_cols + encoded_cols
importances = dict(zip(all_features, model.feature_importances_))

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Model comparison (just one model here for demo)
model_accuracy = {
    "Random Forest": accuracy
}

# Save pipeline
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Save results for Streamlit
results = {
    "feature_importance": importances,
    "roc_curve": (fpr, tpr, roc_auc),
    "model_accuracy": model_accuracy
}

with open("model_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("Model and results saved successfully!")
