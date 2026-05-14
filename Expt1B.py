import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------
# 1. Create Features
# -----------------------
attendance = np.linspace(1, 10, 100)
study_hours = np.linspace(2, 8, 100)
sleep_hours = np.random.normal(7, 1, 100)  # realistic variation

# -----------------------
# 2. Create Target Variable
# -----------------------
noise = np.random.normal(0, 2, 100)

final_marks = (
    5 * attendance +
    4 * study_hours +
    2 * sleep_hours +
    noise
)

# -----------------------
# 3. Create DataFrame
# -----------------------
df = pd.DataFrame({
    "Attendance": attendance,
    "Study_Hours": study_hours,
    "Sleep_Hours": sleep_hours,
    "Final_Marks": final_marks
})

print(df.head())

# -----------------------
# 4. Define X and y
# -----------------------
X = df[["Attendance", "Study_Hours", "Sleep_Hours"]]
y = df["Final_Marks"]

# -----------------------
# 5. Train-Test Split
# -----------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# 6. Train Model
# -----------------------
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------
# 7. Predictions
# -----------------------
y_pred = model.predict(X_test)

# -----------------------
# 8. Evaluation
# -----------------------
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# -----------------------
# 9. Model Coefficients
# -----------------------
print("\nIntercept:", model.intercept_)
print("Coefficients:", model.coef_)

plt.figure(figsize=(10,6))

plt.scatter(y_test, y_pred, color='green', alpha=0.6)

# perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linewidth=2
)

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("AFTER MODEL: Actual vs Predicted")
plt.grid(True)

plt.show()

