import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =============================
# Configuration
# =============================
CSV_PATH = "datasets/dufry_events_activity_unpacked.csv"
ID_COL_CANDIDATES = {"actor_id", "user_id", "id"}
N_TOP_FEATURES = 15
N_PCS_TO_REPORT = 10
SCREE_LIMIT = 20  # limit PCs shown in combined scree plot


# =============================
# Load Data
# =============================
df = pd.read_csv(CSV_PATH)

# Keep numeric columns only
num_df = df.select_dtypes(include=[np.number]).copy()

# Drop ID-like columns if numeric
drop_ids = [c for c in num_df.columns if c.lower() in ID_COL_CANDIDATES]
if drop_ids:
    num_df = num_df.drop(columns=drop_ids)

if num_df.shape[1] < 2:
    raise ValueError("Need at least 2 numeric feature columns for PCA.")

feature_names = num_df.columns.tolist()

# =============================
# Standardization
# =============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(num_df.values)

# =============================
# PCA
# =============================
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)

# =============================
# Print Explained Variance Table
# =============================
k = min(N_PCS_TO_REPORT, len(evr))

report = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(k)],
    "ExplainedVarianceRatio": evr[:k],
    "CumulativeVariance": cum[:k],
})

pd.set_option("display.float_format", lambda x: f"{x:0.6f}")
print("\nExplained variance (first PCs):")
print(report.to_string(index=False))


# =============================
# Standard Scree Plot (Line)
# =============================
pcs = np.arange(1, len(evr) + 1)

plt.figure(figsize=(10, 5))
plt.plot(pcs, evr, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot (Explained Variance per PC)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =============================
# Cumulative Explained Variance Plot
# =============================
plt.figure(figsize=(10, 5))
plt.plot(pcs, cum, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.ylim(0, 1.01)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =============================
# Combined Scree Plot (Bar + Cumulative Line)
# =============================
def scree_plot(pca, annotate=False, limit=None, figsize=(14, 6)):
    values = pca.explained_variance_ratio_

    if limit:
        values = values[:limit]

    n_components = len(values)
    ind = np.arange(1, n_components + 1)  # start at 1
    cumvalues = np.cumsum(values)

    plt.figure(figsize=figsize)
    ax = plt.subplot(111)

    # Convert to percentage
    values_pct = values * 100
    cum_pct = cumvalues * 100

    ax.bar(ind, values_pct, alpha=0.6, label="Individual Explained Variance (%)")
    ax.plot(ind, cum_pct, marker="o", color="red", label="Cumulative Explained Variance (%)")

    if annotate:
        for i in range(n_components):
            ax.annotate(f"{values_pct[i]:.1f}%",
                        (ind[i], values_pct[i]),
                        va="bottom",
                        ha="center",
                        fontsize=9)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    ax.set_xticks(ind)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3)

    if limit:
        plt.title(f"Explained Variance (First {limit} Components)", fontsize=14)
    else:
        plt.title("Explained Variance per Principal Component", fontsize=14)

    plt.tight_layout()
    plt.show()


# Call combined scree plot
scree_plot(pca, annotate=True, limit=SCREE_LIMIT)


# =============================
# Dominant Features per PC
# =============================
components = pca.components_

print("\nTop dominant features per PC (by absolute loading):")

for i in range(k):
    loadings = components[i]
    idx = np.argsort(np.abs(loadings))[::-1][:N_TOP_FEATURES]

    rows = []
    for j in idx:
        rows.append({
            "feature": feature_names[j],
            "loading": loadings[j],
            "abs_loading": abs(loadings[j])
        })

    top_df = pd.DataFrame(rows)
    print(f"\nPC{i+1} (explained variance ratio = {evr[i]:0.6f}):")
    print(top_df.to_string(index=False))


# =============================
# Overall Feature Dominance Score
# =============================
dominance = np.sum(np.abs(components[:k].T) * evr[:k], axis=1)

overall = pd.DataFrame({
    "feature": feature_names,
    "dominance_score": dominance
}).sort_values("dominance_score", ascending=False)

print(f"\nOverall dominant features across first {k} PCs (weighted by EVR):")
print(overall.head(N_TOP_FEATURES).to_string(index=False))


# =============================
# Print Principal Components 1–20 (Loadings)
# =============================

N_PRINT = min(20, len(pca.components_))

print("\nPrincipal Components 1–20 Loadings:\n")

for i in range(N_PRINT):
    pc_loadings = pd.DataFrame({
        "Feature": feature_names,
        "Loading": pca.components_[i],
        "AbsLoading": np.abs(pca.components_[i])
    }).sort_values("AbsLoading", ascending=False)

    print(f"\n==============================")
    print(f"PC{i+1} (Explained Variance: {evr[i]*100:.2f}%)")
    print("==============================")
    print(pc_loadings.head(15).to_string(index=False))


    pc_summary = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(N_PRINT)],
        "ExplainedVariance(%)": evr[:N_PRINT] * 100,
        "CumulativeVariance(%)": np.cumsum(evr[:N_PRINT]) * 100
    })

print("\nPC Summary (1–20):")
print(pc_summary.to_string(index=False))