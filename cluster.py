import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import StandardScaler

# Synthetic Dataset
np.random.seed(42)

n = 200

income = np.random.randint(20, 120, n)
spending = np.random.randint(1, 100, n)

df = pd.DataFrame({
    "Income": income,
    "Spending": spending
})

print(df.head())


X = df[["Income", "Spending"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# K-MEANS
# =========================================================

k = 4

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

centroids = scaler.inverse_transform(
    kmeans.cluster_centers_
)

plt.figure(figsize=(10,6))

sns.scatterplot(
    x="Income",
    y="Spending",
    hue="KMeans_Cluster",
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

plt.title("K-Means Clustering")
plt.legend()
plt.show()


# =========================================================
# EM / GAUSSIAN MIXTURE MODEL
# =========================================================

gmm = GaussianMixture(
    n_components=k,
    random_state=42
)

gmm.fit(X_scaled)
df["GMM_Cluster"] = gmm.predict(X_scaled)

gmm_centers = scaler.inverse_transform(
    gmm.means_
)

plt.figure(figsize=(10,6))

sns.scatterplot(
    x="Income",
    y="Spending",
    hue="GMM_Cluster",
    data=df,
    palette="Set1",
    s=80
)

plt.scatter(
    gmm_centers[:,0],
    gmm_centers[:,1],
    c="black",
    s=250,
    marker="X",
    label="Gaussian Centers"
)

plt.title("EM Clustering using Gaussian Mixture")
plt.legend()
plt.show()