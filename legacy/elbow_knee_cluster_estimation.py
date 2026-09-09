import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator

PCA_FILE = "users_behavior_vectors_pca.csv"
USER_COL = "actor_id"

df = pd.read_csv(PCA_FILE)

if USER_COL in df.columns:
    X = df.drop(columns=[USER_COL]).values
else:
    X = df.values

k_range = range(2, 11)
inertias = []

for k in k_range:
    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    model.fit(X)
    inertias.append(model.inertia_)

# Detect elbow automatically
kneedle = KneeLocator(
    list(k_range),
    inertias,
    curve="convex",
    direction="decreasing"
)

optimal_k = kneedle.knee

plt.figure()
plt.plot(k_range, inertias, marker="o")
if optimal_k:
    plt.axvline(optimal_k, linestyle="--")
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig("elbow_analysis.png", dpi=200)
plt.show()

print("Elbow suggests k =", optimal_k)
