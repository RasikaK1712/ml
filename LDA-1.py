import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)

X_lda = lda.fit_transform(X, y)

# Plot
plt.scatter(X_lda[:,0], X_lda[:,1], c=y)

plt.xlabel("LD1")
plt.ylabel("LD2")

plt.title("LDA on Iris Dataset")

plt.show()