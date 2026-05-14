import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# 1) Synthetic Dataset
# -------------------------
np.random.seed(42)

n = 300

attendance = np.random.randint(1, 11, n)
study_hours = np.random.randint(1, 10, n)
sleep_hours = np.random.randint(4, 10, n)

noise = np.random.normal(0, 3, n)

score = 6*attendance + 5*study_hours + 2*sleep_hours + noise
result = (score >= 60).astype(int)

df = pd.DataFrame({
    "Attendance": attendance,
    "Study_Hours": study_hours,
    "Sleep_Hours": sleep_hours,
    "Result": result
})

# -------------------------
# 2) Features & Target
# -------------------------
X = df[["Attendance", "Study_Hours", "Sleep_Hours"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 4) Bagging Model
# -------------------------
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=50,
    bootstrap=True,
    random_state=42
)

bagging.fit(X_train, y_train)

y_pred_bag = bagging.predict(X_test)

# -------------------------
# 5) Accuracy Results
# -------------------------
print("Bagging Accuracy:", accuracy_score(y_test, y_pred_bag))

print("\nBagging Report:\n")
print(classification_report(y_test, y_pred_bag))

#compare it with a single Decision Tree and plot bar plot for comparison
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Plot comparison
results = pd.DataFrame({
    "Model": ["Decision Tree", "Bagging"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_bag)
    ]
})

plt.figure(figsize=(6,5))
sns.barplot(data=results, x="Model", y="Accuracy", palette="viridis")

plt.title("Decision Tree vs Bagging Accuracy")
plt.ylim(0, 1)

for i, v in enumerate(results["Accuracy"]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

plt.show()