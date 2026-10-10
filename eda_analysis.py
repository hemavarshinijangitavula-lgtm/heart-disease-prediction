import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart.csv")

# Age vs MaxHR (Blob)
plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df,
    x="Age",
    y="MaxHR",
    hue="HeartDisease",
    palette="coolwarm"
)
plt.title("Age vs MaxHR (Heart Disease Blob)")
plt.show()

# Age Distribution
plt.figure(figsize=(7,5))
sns.histplot(
    data=df,
    x="Age",
    hue="HeartDisease",
    bins=30,
    kde=True
)
plt.title("Age Distribution vs Heart Disease")
plt.show()

# Cholesterol Boxplot
plt.figure(figsize=(6,5))
sns.boxplot(
    data=df,
    x="HeartDisease",
    y="Cholesterol"
)
plt.title("Cholesterol vs Heart Disease")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Feature Correlation Heatmap")
plt.show()
