import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from kneed import KneeLocator
import skfuzzy as fuzz

# ============================================================
# Configuration
# ============================================================
PCA_FILE = "users_behavior_vectors_pca.csv"
USER_COL = "actor_id"
K_RANGE = range(2, 11)   # test k = 2..10
RANDOM_STATE = 42
FCM_M = 2.0              # fuzziness coefficient
FCM_ERROR = 1e-5
FCM_MAXITER = 1000

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(PCA_FILE)

if USER_COL in df.columns:
    X = df.drop(columns=[USER_COL]).values
else:
    X = df.values

# remove rows with NaN/inf just in case
X = np.asarray(X, dtype=float)
mask = np.all(np.isfinite(X), axis=1)
X = X[mask]

if X.shape[0] == 0:
    raise ValueError("No valid rows found after removing NaN/inf values.")

# skfuzzy.cmeans expects shape = (n_features, n_samples)
X_fcm = X.T

# ============================================================
# Containers for metrics
# ============================================================
k_values = []
pcs = []
ces = []
dbis = []
inertias = []

# ============================================================
# Compute metrics for each k
# ============================================================
for k in K_RANGE:
    print(f"Processing k={k}...")

    # ----------------------------
    # 1) Fuzzy C-Means for PC, CE
    # ----------------------------
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=X_fcm,
        c=k,
        m=FCM_M,
        error=FCM_ERROR,
        maxiter=FCM_MAXITER,
        init=None,
        seed=RANDOM_STATE
    )

    # u shape: (k, n_samples)
    # Partition Coefficient (PC)
    pc = np.sum(u ** 2) / u.shape[1]

    # Classification Entropy (CE)
    eps = 1e-12
    ce = -np.sum(u * np.log(u + eps)) / u.shape[1]

    # ----------------------------
    # 2) K-Means for DBI and EM
    # ----------------------------
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X)

    inertia = kmeans.inertia_
    dbi = davies_bouldin_score(X, labels)

    # ----------------------------
    # Save metrics
    # ----------------------------
    k_values.append(k)
    pcs.append(pc)
    ces.append(ce)
    dbis.append(dbi)
    inertias.append(inertia)

# ============================================================
# Detect best k values
# ============================================================
# Elbow detection on inertia
kneedle = KneeLocator(
    k_values,
    inertias,
    curve="convex",
    direction="decreasing"
)
optimal_k_elbow = kneedle.knee

# Best k by each metric
best_k_pc = k_values[int(np.argmax(pcs))]       # maximize PC
best_k_ce = k_values[int(np.argmin(ces))]       # minimize CE
best_k_dbi = k_values[int(np.argmin(dbis))]     # minimize DBI

# ============================================================
# Save summary table
# ============================================================
summary_df = pd.DataFrame({
    "k": k_values,
    "Partition_Coefficient_PC": pcs,
    "Classification_Entropy_CE": ces,
    "Davies_Bouldin_Index_DBI": dbis,
    "Elbow_Inertia": inertias
})

summary_df.to_csv("cluster_validation_summary.csv", index=False)

print("\n===== Suggested number of clusters =====")
print(f"Best k by PC   (max): {best_k_pc}")
print(f"Best k by CE   (min): {best_k_ce}")
print(f"Best k by DBI  (min): {best_k_dbi}")
print(f"Best k by Elbow      : {optimal_k_elbow}")

# ============================================================
# Plot 2x2 grid of outcomes with red dotted lines
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

# ----------------------------
# Plot PC
# ----------------------------
axes[0].plot(k_values, pcs, marker="o")
axes[0].axvline(best_k_pc, linestyle=":", color="red", linewidth=2)
axes[0].set_title(f"Partition Coefficient (best k = {best_k_pc})")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("PC")
axes[0].grid(True, alpha=0.3)

# ----------------------------
# Plot CE
# ----------------------------
axes[1].plot(k_values, ces, marker="o")
axes[1].axvline(best_k_ce, linestyle=":", color="red", linewidth=2)
axes[1].set_title(f"Classification Entropy (best k = {best_k_ce})")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("CE")
axes[1].grid(True, alpha=0.3)

# ----------------------------
# Plot DBI
# ----------------------------
axes[2].plot(k_values, dbis, marker="o")
axes[2].axvline(best_k_dbi, linestyle=":", color="red", linewidth=2)
axes[2].set_title(f"Davies-Bouldin Index (best k = {best_k_dbi})")
axes[2].set_xlabel("Number of clusters (k)")
axes[2].set_ylabel("DBI")
axes[2].grid(True, alpha=0.3)

# ----------------------------
# Plot Elbow / Inertia
# ----------------------------
axes[3].plot(k_values, inertias, marker="o")

if optimal_k_elbow is not None:
    axes[3].axvline(optimal_k_elbow, linestyle=":", color="red", linewidth=2)
    title_elbow = f"Elbow Method (suggested k = {optimal_k_elbow})"
else:
    title_elbow = "Elbow Method (no clear elbow detected)"

axes[3].set_title(title_elbow)
axes[3].set_xlabel("Number of clusters (k)")
axes[3].set_ylabel("Inertia")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cluster_validation_grid.png", dpi=300, bbox_inches="tight")
plt.show()