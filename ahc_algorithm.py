import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

INPUT_CSV = Path('datasets/leonardo_activity_unpacked_step3_log1p_winsor_robust_1_99.csv')
OUTPUT_ROOT = Path('aglo_HC_1_12')
RANDOM_SEED = 42
K_RANGE = range(2, 13)
MAX_PLOT_POINTS = 200
SILHOUETTE_SAMPLE = 1500
STABILITY_SAMPLE = 1200
N_BOOTSTRAPS = 4
BOOTSTRAP_FRAC = 0.8
TOP_FEATURES = 5
METHOD = 'Agglomerative Hierarchical Clustering'
LINKAGE = 'ward'


def load_data(path: Path):
    df = pd.read_csv(path)
    id_col = 'actor_id' if 'actor_id' in df.columns else df.columns[0]
    feature_cols = [c for c in df.columns if c != id_col]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    return df, id_col, feature_cols, X


def compute_centroids(X, labels, k):
    return np.vstack([X[labels == c].mean(axis=0) for c in range(1, k + 1)])


def size_balance_score(labels, k):
    counts = np.bincount(labels, minlength=k + 1)[1:].astype(float)
    p = counts / counts.sum()
    uniform = np.ones(k) / k
    l1 = np.abs(p - uniform).sum()
    max_l1 = 2 * (1 - 1 / k)
    return float(max(0.0, 1.0 - l1 / max_l1))


def centroid_separation_score(centroids):
    dists = [np.linalg.norm(centroids[i] - centroids[j]) for i, j in combinations(range(len(centroids)), 2)]
    mean_dist = float(np.mean(dists)) if dists else 0.0
    return float(1.0 - np.exp(-mean_dist / 2.0))


def feature_dominance_score(centroids):
    gaps = []
    for row in np.abs(centroids):
        s = np.sort(row)[::-1]
        denom = s[0] + 1e-12
        gap = 1.0 if len(s) == 1 else np.clip((s[0] - s[1]) / denom, 0.0, 1.0)
        gaps.append(float(gap))
    return float(np.mean(gaps))


def compute_interpretability(X, labels, k):
    centroids = compute_centroids(X, labels, k)
    balance = size_balance_score(labels, k)
    separation = centroid_separation_score(centroids)
    dominance = feature_dominance_score(centroids)
    score = 0.50 * dominance + 0.30 * separation + 0.20 * balance
    return float(score), {
        'feature_dominance': dominance,
        'centroid_separation': separation,
        'size_balance': balance
    }, centroids


def sample_indices(n, max_n, rng):
    if n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def silhouette_on_sample(X, labels, rng):
    idx = sample_indices(len(X), SILHOUETTE_SAMPLE, rng)
    Xs = X[idx]
    ys = labels[idx]
    if len(np.unique(ys)) < 2:
        return float('nan')
    return float(silhouette_score(Xs, ys, metric='euclidean'))


def stability_via_subsample(X, k, rng):
    base_idx = sample_indices(len(X), STABILITY_SAMPLE, rng)
    Xb = X[base_idx]
    sample_size = max(k + 2, int(round(len(Xb) * BOOTSTRAP_FRAC)))
    clusterings = []

    for _ in range(N_BOOTSTRAPS):
        idx_local = np.sort(rng.choice(len(Xb), size=sample_size, replace=False))
        Z = linkage(Xb[idx_local], method=LINKAGE, metric='euclidean', optimal_ordering=False)
        labels_local = fcluster(Z, t=k, criterion='maxclust')
        clusterings.append((idx_local, labels_local))

    aris = []
    for i, j in combinations(range(len(clusterings)), 2):
        idx_i, lab_i = clusterings[i]
        idx_j, lab_j = clusterings[j]
        common, pos_i, pos_j = np.intersect1d(idx_i, idx_j, return_indices=True)
        if len(common) > k:
            aris.append(adjusted_rand_score(lab_i[pos_i], lab_j[pos_j]))

    return float(np.mean(aris)) if aris else float('nan')


def save_visualization(X, labels, centroids, out_png, title, rng):
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X2 = pca.fit_transform(X)
    C2 = pca.transform(centroids)

    idx = sample_indices(len(X), MAX_PLOT_POINTS, rng)
    X2p = X2[idx]
    yp = labels[idx]

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    scatter = ax.scatter(
        X2p[:, 0],
        X2p[:, 1],
        c=yp,
        s=32,
        alpha=0.65
    )

    ax.scatter(
        C2[:, 0],
        C2[:, 1],
        marker='*',
        s=320,
        c='gold',
        edgecolors='black',
        linewidths=1.0,
        label='Cluster centers'
    )

    ax.set_title(title)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.grid(alpha=0.25)

    leg1 = ax.legend(*scatter.legend_elements(), title='Cluster', loc='best', frameon=True)
    ax.add_artist(leg1)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def top_features_df(centroids, feature_cols):
    rows = []
    for c, row in enumerate(centroids, start=1):
        idx = np.argsort(np.abs(row))[::-1][:TOP_FEATURES]
        rec = {'cluster': c}
        for r, j in enumerate(idx, start=1):
            rec[f'feature_{r}'] = feature_cols[j]
            rec[f'value_{r}'] = float(row[j])
        rows.append(rec)
    return pd.DataFrame(rows)


def minmax(s):
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if np.isclose(mx, mn):
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    df, id_col, feature_cols, X_raw = load_data(INPUT_CSV)
    X = StandardScaler().fit_transform(X_raw)

    print(f'Loaded data: {X.shape[0]} rows x {X.shape[1]} features')
    print('Building Ward linkage tree on full dataset...')
    Z_full = linkage(X, method=LINKAGE, metric='euclidean', optimal_ordering=False)
    print('[OK] full linkage tree built')

    summary_rows = []

    for k in K_RANGE:
        print(f'Processing k={k}...')
        out_dir = OUTPUT_ROOT / f'k_{k}'
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = fcluster(Z_full, t=k, criterion='maxclust')
        sil = silhouette_on_sample(X, labels, rng)
        stab = stability_via_subsample(X, k, rng)
        interp, interp_parts, centroids = compute_interpretability(X, labels, k)

        pd.DataFrame({
            id_col: df[id_col],
            'cluster': labels
        }).to_csv(out_dir / 'assignments.csv', index=False)

        sizes = pd.Series(labels).value_counts().sort_index()
        pd.DataFrame({
            'cluster': sizes.index,
            'size': sizes.values,
            'proportion': sizes.values / sizes.sum()
        }).to_csv(out_dir / 'cluster_sizes.csv', index=False)

        cdf = pd.DataFrame(centroids, columns=feature_cols)
        cdf.insert(0, 'cluster', np.arange(1, k + 1))
        cdf.to_csv(out_dir / 'cluster_centroids_standardized.csv', index=False)

        top_features_df(centroids, feature_cols).to_csv(out_dir / 'cluster_top_features.csv', index=False)

        save_visualization(
            X,
            labels,
            centroids,
            out_dir / 'visualization_pca_2d.png',
            f'Agglomerative HC (k={k}) in PCA space',
            rng
        )

        (out_dir / 'metrics.json').write_text(json.dumps({
            'k': k,
            'method': METHOD,
            'linkage': LINKAGE,
            'silhouette_fsi': sil,
            'cluster_stability': stab,
            'interpretability_score': interp,
            'interpretability_components': interp_parts,
            'plot_point_limit': MAX_PLOT_POINTS,
            'silhouette_sample_size': min(len(X), SILHOUETTE_SAMPLE),
            'stability_sample_size': min(len(X), STABILITY_SAMPLE),
        }, indent=2))

        summary_rows.append({
            'k': k,
            'method': 'Agglomerative HC',
            'silhouette_fsi': sil,
            'cluster_stability': stab,
            'interpretability_score': interp,
            'size_balance': interp_parts['size_balance'],
            'centroid_separation': interp_parts['centroid_separation'],
            'feature_dominance': interp_parts['feature_dominance'],
        })

    summary = pd.DataFrame(summary_rows)

    summary['silhouette_norm'] = minmax(summary['silhouette_fsi'])
    summary['stability_norm'] = minmax(summary['cluster_stability'])
    summary['interpretability_norm'] = minmax(summary['interpretability_score'])

    summary['overall_score'] = (
            0.40 * summary['silhouette_norm'] +
            0.30 * summary['cluster_stability'] +
            0.30 * summary['interpretability_score']
    )

    summary = summary.sort_values(
        by=['overall_score', 'silhouette_fsi', 'cluster_stability', 'interpretability_score'],
        ascending=False
    ).reset_index(drop=True)

    summary.insert(0, 'rank', np.arange(1, len(summary) + 1))
    summary.to_csv(OUTPUT_ROOT / 'summary_ranked.csv', index=False)

    best_k = int(summary.loc[0, 'k'])

    (OUTPUT_ROOT / 'README.txt').write_text(
        '\n'.join([
            'Agglomerative Hierarchical Clustering outputs for k=2..12.',
            f'Input dataset: {INPUT_CSV.name}',
            f'Rows: {len(df)}',
            f'Features: {len(feature_cols)}',
            f'Best k by composite ranking: {best_k}',
            '',
            'Composite ranking = 40% normalized silhouette + 30% stability + 30% interpretability.',
            'Interpretability = 50% feature dominance + 30% centroid separation + 20% size balance.',
            f'Visualizations are PCA projections capped at {MAX_PLOT_POINTS} points.',
            f'Silhouette is estimated on up to {SILHOUETTE_SAMPLE} points for efficiency.',
            f'Stability is estimated on up to {STABILITY_SAMPLE} points using subsampled ARI.',
        ])
    )

    print(summary.to_string(index=False))
    print(f'[OK] wrote outputs to {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()