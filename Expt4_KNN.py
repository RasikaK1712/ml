# 1) Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2) Synthetic Data
np.random.seed(42)

n = 200

attendance = np.random.randint(1, 11, n)
study_hours = np.random.randint(1, 10, n)
sleep_hours = np.random.randint(4, 10, n)

noise = np.random.normal(0, 3, n)

# 3) Target rule
score = 6 * attendance + 5 * study_hours + 2 * sleep_hours + noise
result = (score >= 60).astype(int)

# 4) DataFrame
df = pd.DataFrame({
    "Attendance": attendance,
    "Study_Hours": study_hours,
    "Sleep_Hours": sleep_hours,
    "Result": result
})

print(df.head())

# ---------------------------------------------------
# ✔️ IMPORTANT FIX: Use ONLY 2 features for plotting
# ---------------------------------------------------
X = df[["Attendance", "Study_Hours"]]
y = df["Result"]

# 5) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7) Model
k = int(np.sqrt(len(X_train)))
model = KNeighborsClassifier(n_neighbors=k)

# 8) Train
model.fit(X_train, y_train)

# 9) Predict
y_pred = model.predict(X_test)

# 10) Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from matplotlib.colors import ListedColormap

def plot_knn_boundary(X, y, model):

    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10,6))

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)

    plt.title("KNN Decision Boundary (Fixed)")
    plt.xlabel("Attendance (scaled)")
    plt.ylabel("Study Hours (scaled)")
    plt.show()


plot_knn_boundary(X_train, y_train, model)