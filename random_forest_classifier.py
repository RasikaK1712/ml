#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


#data
data = {
    "Age": [22, 25, 28, 35, 42, 30, 45, 50, 27, 38],
    
    "EstimatedSalary": [
        25000, 32000, 48000, 65000, 85000,54000, 92000, 110000, 73000, 67000],
    
    "Purchased": [0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
}
df=pd.DataFrame(data)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#train test split
X_train, X_test, y_train , y_test = train_test_split(X,y,test_size = 0.3)

#train model
model=RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

#predictions
y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

#metrics
train_accuracy=accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)
print(f'Training Accuracy = {train_accuracy}')
print(f'Test Accuracy = {test_accuracy}')
print('Confusion Matrix')
print(cm)