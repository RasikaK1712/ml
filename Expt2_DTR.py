import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# 1) Create Features
attendance = np.random.randint(1,10,50)
study_hours = np.random.randint(2,10,50)
sleep_hours = np.random.randint(4,10,50)

# 2) Add noise
noise = np.random.normal(0, 2, 50)

# 3) Continuous Target (REGRESSION)
final_marks = (
    5 * attendance +
    4 * study_hours +
    2 * sleep_hours +
    noise
)

# 4) Create DataFrame
df = pd.DataFrame({
    "Attendance": attendance,
    "Study_Hours": study_hours,
    "Sleep_Hours": sleep_hours,
    "Final_Marks": final_marks
})

print(df.head())

# 5) Features & Target
X = df[["Attendance", "Study_Hours", "Sleep_Hours"]]
y = df["Final_Marks"]

# 6) Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7) Regression Model
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    criterion='squared_error',
    # max_depth=3,
    random_state=42
)

# 8) Train
model.fit(X_train, y_train)

# 9) Predict
y_pred = model.predict(X_test)

# 10) Evaluation
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMSE: {mse}")
print(f"R2 Score: {r2}")

from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True
)

plt.title("Decision Tree Regressor")
plt.show()