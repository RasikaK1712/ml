#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#data
data = {
    "Age": [22, 25, 28, 35, 42, 30, 45, 50, 27, 38],
    
    "EstimatedSalary": [
        25000, 32000, 48000, 65000, 85000,
        54000, 92000, 110000, 73000, 67000
    ],
    
    "Purchased": [0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
}
df=pd.DataFrame(data)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#train test split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size = 0.3)

#feature scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#finding optimal k
k_values=range(1,6)
accuracy=np.zeros(5)
for k in k_values:
    model=KNeighborsClassifier(n_neighbors=k, metric= 'euclidean')
    model.fit(X_train, y_train)
    y_test_pred=model.predict(X_test)
    accuracy[k-1]=accuracy_score(y_test, y_test_pred)
    print(f'Accuracy of model with k = {k} is : {accuracy[k-1]}')

best_accuracy=np.max(accuracy)

for k in k_values:
    if accuracy[k-1] == best_accuracy:
        optimal_k=k
        exit
    else: 
        continue

#plot accuracy vs k
plt.scatter(k_values, accuracy, color = 'blue')
plt.plot(k_values,accuracy, color = 'red')
plt.xlabel('K values')
plt.ylabel('Accuracy Score')
plt.title('Accuracy v/s K')
plt.show()

#train final model
model=KNeighborsClassifier(n_neighbors = optimal_k, metric = 'euclidean')
model.fit(X_train, y_train)

#predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy =accuracy_score(y_test, y_test_pred)
cm=confusion_matrix(y_test, y_test_pred)
print(f'Training Accuracy = {train_accuracy}')
print(f'Test Accuracy = {test_accuracy}')
print('Confusion Matrix')
print(cm)
