#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


#data
data = {
    "Age": [22, 25, 28, 30, 35, 40, 45, 50, 32, 38],

    "Experience": [1, 2, 4, 5, 8, 10, 15, 20, 6, 9],

    "ProjectsCompleted": [2, 3, 5, 6, 9, 12, 15, 18, 7, 11],

    "Salary": [
        30000, 35000, 45000, 50000, 65000,
        80000, 95000, 120000, 55000, 72000
    ]
}
df=pd.DataFrame(data)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#train test split
X_train, X_test, y_train , y_test=train_test_split(X, y , test_size = 0.3)

#model
model=RandomForestRegressor(n_estimators = 10)
model.fit(X_train, y_train)

#predictions
y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

#metrics
train_accuracy=r2_score(y_train,y_train_pred)
test_accuracy=r2_score(y_test,y_test_pred)
training_loss=mean_squared_error(y_train,y_train_pred)
testing_loss=mean_squared_error(y_test,y_test_pred)
print(f'Training Accuracy = {train_accuracy}')
print(f'Testing Accuracy = {test_accuracy}')
print(f'Training Loss = {training_loss}')
print(f'Testing Loss = {testing_loss}')