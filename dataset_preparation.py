import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


# -----------------------------
# 1) Load
# -----------------------------
DATA_PATH = "datasets/leonardo_unpacked.csv"
df = pd.read_csv(DATA_PATH)

# Privacy: keep user key separate (do not use for modeling)
USER_COL = "actor_id"
if USER_COL not in df.columns:
    raise ValueError(f"Expected a user identifier column named '{USER_COL}'.")

user_ids = df[USER_COL].copy()

# Identify event/count columns (numeric features only)
count_cols = [c for c in df.columns if c != USER_COL and pd.api.types.is_numeric_dtype(df[c])]
if not count_cols:
    raise ValueError("No numeric activity columns found to build behavioral vectors.")

X_counts = df[count_cols].fillna(0).clip(lower=0)


# -----------------------------
# 2) Privacy-preserving aggregated behavioral user vectors
#    (already user-level rows; we further engineer robust, non-identifying features)
# -----------------------------
def safe_divide(a, b, eps=1e-12):
    return a / (b + eps)

# If `total_events` exists, use it; otherwise compute it
if "total_events" in X_counts.columns:
    total = X_counts["total_events"].astype(float).values
    event_cols = [c for c in count_cols if c != "total_events"]
else:
    event_cols = count_cols
    total = X_counts[event_cols].sum(axis=1).astype(float).values

# --- Define event groups (best-effort, based on column names) ---
# You can refine these mappings to match your platform semantics.
creation_patterns = [
    "post-created", "post-published", "comment-created", "message-created",
    "group-chat-created", "member-added-to-chat", "admin-role-granted"
]
deletion_patterns = [
    "post-deleted", "comment-deleted", "message-deleted", "admin-role-revoked",
    "member-removed-from-chat"
]
consumption_patterns = [
    "post-read", "post-read-updated", "messages-read", "push-notification-sent",
    "unread-inbox-count-changed"
]
reaction_patterns = [
    "reaction", "post-reaction-created", "post-reaction-updated",
    "post-reaction-deleted", "comment-liked", "comment-unliked"
]
moderation_patterns = [
    "post-reported", "comment-reported", "post-moved", "group-chat-details-updated"
]

def sum_by_patterns(frame, patterns):
    cols = [c for c in frame.columns if c in patterns]
    if not cols:
        return np.zeros(len(frame), dtype=float)
    return frame[cols].sum(axis=1).astype(float).values

created = sum_by_patterns(X_counts, creation_patterns)
deleted = sum_by_patterns(X_counts, deletion_patterns)
consumed = sum_by_patterns(X_counts, consumption_patterns)
reacted = sum_by_patterns(X_counts, reaction_patterns)
moderated = sum_by_patterns(X_counts, moderation_patterns)

# -----------------------------
# 3) Engineer statistically robust engagement features
# -----------------------------
features = pd.DataFrame(index=df.index)

# Volume / intensity
features["total_events"] = total
features["nonzero_event_types"] = (X_counts[event_cols] > 0).sum(axis=1).astype(float).values
features["avg_events_per_active_type"] = safe_divide(total, features["nonzero_event_types"].values)

# Role-like behavioral composition (privacy-preserving, aggregate)
features["created_events"] = created
features["deleted_events"] = deleted
features["consumed_events"] = consumed
features["reacted_events"] = reacted
features["moderated_events"] = moderated

features["share_created"] = safe_divide(created, total)
features["share_deleted"] = safe_divide(deleted, total)
features["share_consumed"] = safe_divide(consumed, total)
features["share_reacted"] = safe_divide(reacted, total)
features["share_moderated"] = safe_divide(moderated, total)

# Activity diversity / entropy across event types (robust for skewed ESN logs)
p = X_counts[event_cols].astype(float).values
row_sums = p.sum(axis=1, keepdims=True)
p = p / (row_sums + 1e-12)
entropy = -np.sum(p * np.log(p + 1e-12), axis=1)  # Shannon entropy
features["event_entropy"] = entropy
features["event_entropy_norm"] = safe_divide(entropy, np.log(len(event_cols) + 1e-12))

# Concentration (Gini-like) using sorted proportions (simple, stable proxy)
# Higher = more concentrated in a few event types.
p_sorted = np.sort(p, axis=1)
n = p_sorted.shape[1]
index = np.arange(1, n + 1)
gini = (np.sum((2 * index - n - 1) * p_sorted, axis=1)) / (n - 1 + 1e-12)
features["event_concentration"] = gini

# Ratio features (interpretability-friendly)
features["create_to_consume"] = safe_divide(created, consumed)
features["react_to_create"] = safe_divide(reacted, created)
features["delete_to_create"] = safe_divide(deleted, created)

# Optional: include per-event shares for additional detail (still privacy-preserving)
# This can improve clustering but increases dimensionality.
for c in event_cols:
    features[f"share_{c}"] = safe_divide(X_counts[c].astype(float).values, total)

# Replace any non-finite values
features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# -----------------------------
# 4) Transform skew + standardize
# -----------------------------
# Log1p stabilizes long-tailed count distributions and ratios.
features_log = features.copy()
for col in features_log.columns:
    # Shares already in [0,1], entropy/concentration small; log1p is safe but optional.
    # Apply log1p to all to keep pipeline simple and robust.
    features_log[col] = np.log1p(np.clip(features_log[col].astype(float), a_min=0, a_max=None))

# Robust scaling (median/IQR) tends to work better than z-score for heavy tails/outliers.
scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
X_scaled = scaler.fit_transform(features_log.values)


# -----------------------------
# 5) Reduce dimensionality (PCA)
# -----------------------------
# Keep enough components to explain ~90% variance (tunable).
pca = PCA(n_components=0.90, svd_solver="full", random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Input users:", len(df))
print("Engineered features:", features_log.shape[1])
print("PCA components:", X_pca.shape[1])
print("Explained variance (sum):", float(pca.explained_variance_ratio_.sum()))


# -----------------------------
# 6) Outputs for clustering
# -----------------------------
# X_pca: ready for K-Means / FCM / PFCM / GK / AHC
# features: interpretable feature table (keep user_ids separate for privacy)
out = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
out.insert(0, USER_COL, user_ids)

# Save artifacts (optional)
out.to_csv("users_behavior_vectors_pca.csv", index=False)
features.to_csv("users_engineered_features_raw.csv", index=False)

# If you need the modeling matrix without identifiers:
# X_model = X_pca
