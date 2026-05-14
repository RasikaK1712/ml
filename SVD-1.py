import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import TruncatedSVD

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Apply SVD
svd = TruncatedSVD(n_components=2)

X_svd = svd.fit_transform(X)

# Plot
plt.scatter(X_svd[:,0], X_svd[:,1], c=y)

plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.title("SVD on Iris Dataset")

plt.show()