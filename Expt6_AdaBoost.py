import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
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

# -------------------------
# 3) Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 4) Scaling (optional but OK for consistency)
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 5) AdaBoost Model
# -------------------------
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # weak learner (stump)
    n_estimators=100,                               # number of weak learners
    learning_rate=1.0,                              # contribution of each learner
    random_state=42
)

# -------------------------
# 6) Train
# -------------------------
ada_model.fit(X_train_scaled, y_train)

# -------------------------
# 7) Predict
# -------------------------
y_pred_ada = ada_model.predict(X_test_scaled)

# -------------------------
# 8) Evaluation
# -------------------------
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))

print("\n--- AdaBoost Classification Report ---")
print(classification_report(y_test, y_pred_ada))