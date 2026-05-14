#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#dataset
X,y= make_blobs(
    n_features=2,
    centers = 3,
    n_samples = 300
)

#plot dataset
plt.scatter(X[:,0], X[:,-1], color = 'blue')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('Original Dataset')
plt.show()

#feature scaling
sc=StandardScaler()
X=sc.fit_transform(X)

#plot dataset
plt.scatter(X[:,0], X[:,-1], color = 'blue')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('Original Dataset after Feature Scaling')
plt.show()

#elbow
inertia = []
k_values=range(1,11)
for k in k_values:
    model=KMeans(n_clusters=k)
    model.fit(X)
    inertia.append(model.inertia_)

#elbow plot
plt.scatter(k_values,inertia, color='blue')
plt.plot(k_values,inertia, color='blue')
plt.xlabel('K')
plt.ylabel('WCSS')
plt.title('Elbow Plot')
plt.show()

#optimal k
optimal_k=int(input('Look at the elbow plot and input optimal value of k : '))

#train model on optimal k
model=KMeans(n_clusters=optimal_k)
model.fit(X)
labels=model.predict(X)

#plot clusters
plt.scatter(X[:,0], X[:,-1], c=labels)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('Final CLusters')
plt.show()

#Silhouette Score
sil=silhouette_score(X,labels)
print(f'Silhouette Score = {sil}')