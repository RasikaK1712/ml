# 1) Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 2) Create Synthetic Data
np.random.seed(42)

n = 200

attendance = np.random.randint(1, 11, n)
study_hours = np.random.randint(1, 10, n)

noise = np.random.normal(0, 3, n)

# 3) Hidden rule (score-based classification)
score = 6 * attendance + 5 * study_hours + noise

# 4) Convert to class
result = (score >= 60).astype(int)

# 5) Create DataFrame
df = pd.DataFrame({
    "Attendance": attendance,
    "Study_Hours": study_hours,
    "Result": result
})

print(df.head())

# 6) Features & Target
X = df[["Attendance", "Study_Hours"]]
y = df["Result"]

# 7) Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8) Feature Scaling (IMPORTANT for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9) SVM Model
model = SVC(kernel='linear', C=1.0, gamma='scale')

# 10) Train
model.fit(X_train, y_train)

# 11) Predict
y_pred = model.predict(X_test)

# 12) Evaluation
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def plot_boundary(X, y, model):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10,6))

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)

    plt.xlabel("Attendance (scaled)")
    plt.ylabel("Study Hours (scaled)")
    plt.title("SVM Decision Boundary (Student Pass/Fail)")
    plt.show()


plot_boundary(X_train, y_train, model)