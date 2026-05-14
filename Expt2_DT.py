import numpy as np
import pandas as pd

np.random.seed(42)

attendance = np.random.randint(1,10,30)
study_hours = np.random.randint(2,10,30)
sleep_hours = np.random.randint(4,10,30)

noise = np.random.normal(0, 2, 30)

final_marks = 5 * attendance + 4 * study_hours + 2 * sleep_hours + noise

result = (final_marks>=60).astype(int)

df= pd.DataFrame({
    "Attendance": attendance, "Study_Hours": study_hours, "Sleep_Hours": sleep_hours, "Final_Marks": final_marks, "Result": result})

print(df.head())

X = df[["Attendance", "Study_Hours", "Sleep_Hours"]]
y = df["Result"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier , plot_tree
model = DecisionTreeClassifier(
    criterion ='gini',max_depth=3, random_state=42, min_samples_split=5, min_samples_leaf=2
)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Fail", "Pass"],
    filled=True,
    # rounded=True
)

plt.show()