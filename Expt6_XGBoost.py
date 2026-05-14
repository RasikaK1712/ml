import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

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
# 4) Scaling (optional for trees, but OK for consistency)
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 5) XGBoost Model (CORRECT)
# -------------------------
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    # use_label_encoder=False,
    eval_metric='logloss'
)

# -------------------------
# 6) Train
# -------------------------
xgb_model.fit(X_train_scaled, y_train)

# -------------------------
# 7) Predict
# -------------------------
y_pred_xgb = xgb_model.predict(X_test_scaled)

# -------------------------
# 8) Evaluation
# -------------------------
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

print("\n--- XGBoost Classification Report ---")
print(classification_report(y_test, y_pred_xgb))