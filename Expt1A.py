import numpy as np
import pandas as pd
np.random.seed(42)

attendance = np.linspace(1, 10, 100)
noise = np.random.normal(0, 2, 100)
final_marks = 5 * attendance + 30 + noise

df = pd.DataFrame({'Attendance': attendance, 'Final Marks': final_marks})

print(df.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.scatter(df['Attendance'], df['Final Marks'], color='blue', label='Data Points')
plt.xlabel("Attendance")
plt.ylabel("Final Marks")
plt.title("Attendance vs Final Marks")
plt.legend()
plt.grid()
plt.show()

from sklearn.linear_model import LinearRegression

X= df['Attendance'].values.reshape(-1,1)
y = df['Final Marks'].values

from sklearn.model_selection import train_test_split
X_train , X_test,y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Predicted Line')
plt.xlabel("Attendance")
plt.ylabel("Final Marks")
plt.title("Attendance vs Final Marks (Test Set)")
plt.legend()
plt.grid()
plt.show()