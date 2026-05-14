import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -------------------------
# 1) Synthetic Dataset
# -------------------------
np.random.seed(42)

n = 200

attendance = np.random.randint(40, 100, n)   # Attendance (%)
study_hours = np.random.randint(0, 10, n)    # Study Hours per day

df = pd.DataFrame({
    "Attendance": attendance,
    "StudyHours": study_hours
})

print(df.head())

# -------------------------
# 2) Features
# -------------------------
X = df[["Attendance", "StudyHours"]]

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
    em = GaussianMixture(n_components=k, random_state=42)
    labels = em.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(7,4))
plt.plot(k_values, scores, marker='o', color='orange')
plt.title("Silhouette Score vs K (EM Clustering)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Score")
plt.grid()
plt.show()

# -------------------------
# 5) Final EM Model
# -------------------------
optimal_k = 3

em_model = GaussianMixture(
    n_components=optimal_k,
    covariance_type='full',
    random_state=42
)

labels = em_model.fit_predict(X_scaled)
df["Cluster"] = labels

# -------------------------
# 6) Gaussian Ellipse Plot (Soft Clustering)
# -------------------------
plt.figure(figsize=(10, 7))
ax = plt.gca()

sns.scatterplot(
    x=X_scaled[:, 0],
    y=X_scaled[:, 1],
    hue=labels,
    palette='viridis',
    s=70,
    ax=ax
)

def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()

    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    for nsig in range(1, 4):
        ax.add_patch(
            Ellipse(
                position,
                nsig * width,
                nsig * height,
                angle=angle,
                **kwargs
            )
        )

for pos, covar, w in zip(
    em_model.means_,
    em_model.covariances_,
    em_model.weights_
):
    draw_ellipse(pos, covar, alpha=w * 0.15, color='blue')

plt.title("EM Clustering: Attendance vs Study Hours (Gaussian Mixtures)")
plt.show()

# -------------------------
# 7) Model Metrics
# -------------------------
print(f"Model Converged: {em_model.converged_}")
print(f"Log-Likelihood: {em_model.score(X_scaled):.4f}")
print(f"AIC Score: {em_model.aic(X_scaled):.2f}")