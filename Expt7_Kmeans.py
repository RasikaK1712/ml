import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -------------------------
# 1) Synthetic Dataset
# -------------------------
np.random.seed(42)

n = 200

income = np.random.randint(20, 120, n)          # Annual Income
spending = np.random.randint(1, 100, n)         # Spending Score

df = pd.DataFrame({
    "Income": income,
    "Spending": spending
})

print(df.head())

# -------------------------
# 2) Features
# -------------------------
X = df[["Income", "Spending"]]

# -------------------------
# 3) Scaling
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 4) Find best K (Silhouette Score)
# -------------------------
scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(7,4))
plt.plot(k_values, scores, marker='o', color='orange')
plt.title("Silhouette Score vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Score")
plt.grid()
plt.show()

# -------------------------
# 5) Final Model
# -------------------------
optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------
# 6) Centroids
# -------------------------
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# -------------------------
# 7) Plot Clusters
# -------------------------
plt.figure(figsize=(10,6))

sns.scatterplot(
    x="Income",
    y="Spending",
    hue="Cluster",
    data=df,
    palette="viridis",
    s=80
)

plt.scatter(
    centroids[:,0],
    centroids[:,1],
    c="red",
    s=250,
    marker="X",
    label="Centroids"
)

plt.title(f"K-Means Clustering (K={optimal_k})")
plt.legend()
plt.show()