#!/usr/bin/env python3
"""
PFCM (Possibilistic Fuzzy C-Means) clustering for noisy / overlapping behavior.

Features:
- Accepts BOTH raw-count datasets and already-preprocessed datasets
- Tries k = 2..12 by default
- Saves per-k memberships/typicalities + centers + plots
- Computes & saves per-k metrics (CSV + JSON), including:
    * Silhouette
    * FSI (fuzzy prototype silhouette)
    * Cluster Stability (mean pairwise ARI across repeated runs)
    * Interpretability Score
    * Objective, Xie-Beni, Partition Coefficient, Classification Entropy
- Saves compact summary CSV:
    pfcm_cluster_quality_summary.csv
  with columns:
    k, silhouette, fsi, cluster_stability_ari, interpretability

Preprocessing modes:
- raw_counts   -> log1p + StandardScaler
- preprocessed -> use numeric values as-is
- auto         -> infer from data:
    * min <= -1.0  -> use as-is
    * -1.0 < min < 0.0 -> StandardScaler only
    * min >= 0.0   -> log1p + StandardScaler

Examples:
  python PFCM_Possibilistic_Algorithm.py \
      --input datasets/leonardo_activity_unpacked.csv \
      --input_mode raw_counts

  python PFCM_Possibilistic_Algorithm.py \
      --input datasets/leonardo_activity_unpacked_step3_log1p_winsor_robust_1_99.csv \
      --input_mode preprocessed
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

import matplotlib.pyplot as plt


EPS = 1e-12


@dataclass
class PFCMResult:
    centers: np.ndarray          # (k, d)
    U: np.ndarray                # (n, k) membership (sum_k U=1 per point)
    T: np.ndarray                # (n, k) typicality (independent per cluster)
    gamma: np.ndarray            # (k,)
    objective: float
    n_iter: int


def _squared_euclidean(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Return D2 where D2[i,k] = ||X[i]-V[k]||^2
    Shapes: X (n,d), V (k,d) -> D2 (n,k)
    """
    diff = X[:, None, :] - V[None, :, :]
    return np.sum(diff * diff, axis=2)


def _update_U_from_distances(D2: np.ndarray, m: float) -> np.ndarray:
    """
    Standard FCM update:
      u_ik = 1 / sum_j ( (d_ik / d_ij)^(2/(m-1)) )
    using squared distances D2 (avoid sqrt).
    """
    n, k = D2.shape
    power = 1.0 / (m - 1.0)

    U = np.zeros((n, k), dtype=float)
    zero_mask = D2 <= EPS

    if np.any(zero_mask):
        for i in range(n):
            z = np.where(zero_mask[i])[0]
            if len(z) > 0:
                U[i, z] = 1.0 / len(z)
            else:
                di = D2[i]
                ratios = (di[None, :] / (di[:, None] + EPS)) ** power
                U[i] = 1.0 / (np.sum(ratios, axis=1) + EPS)
        return U

    ratios = (D2[:, :, None] / (D2[:, None, :] + EPS)) ** power
    denom = np.sum(ratios, axis=2)
    U = 1.0 / (denom + EPS)
    U = U / (np.sum(U, axis=1, keepdims=True) + EPS)
    return U


def _update_gamma(
        D2: np.ndarray,
        U: np.ndarray,
        T: np.ndarray,
        a: float,
        b: float,
        m: float,
        eta: float,
) -> np.ndarray:
    """
    Cluster scale parameters gamma_k:
      gamma_k = sum_i w_ik * d2_ik / sum_i w_ik
      w_ik = a*u_ik^m + b*t_ik^eta
    """
    W = a * (U ** m) + b * (T ** eta)
    num = np.sum(W * D2, axis=0)
    den = np.sum(W, axis=0) + EPS
    gamma = num / den
    gamma = np.maximum(gamma, EPS)
    return gamma


def _update_T_from_distances(
        D2: np.ndarray,
        gamma: np.ndarray,
        eta: float,
        b: float,
) -> np.ndarray:
    """
    PCM-like typicality update:
      t_ik = 1 / (1 + (b/gamma_k)*d2_ik)^(1/(eta-1))
    """
    exponent = 1.0 / (eta - 1.0)
    frac = (b * D2) / (gamma[None, :] + EPS)
    T = 1.0 / ((1.0 + frac) ** exponent)
    return np.clip(T, 0.0, 1.0)


def _update_centers(
        X: np.ndarray,
        U: np.ndarray,
        T: np.ndarray,
        a: float,
        b: float,
        m: float,
        eta: float,
) -> np.ndarray:
    """
    Center update:
      v_k = sum_i w_ik x_i / sum_i w_ik
      w_ik = a*u_ik^m + b*t_ik^eta
    """
    W = a * (U ** m) + b * (T ** eta)
    num = W.T @ X
    den = np.sum(W, axis=0)[:, None] + EPS
    return num / den


def _objective(
        D2: np.ndarray,
        U: np.ndarray,
        T: np.ndarray,
        a: float,
        b: float,
        m: float,
        eta: float,
) -> float:
    """
    PFCM objective:
      J = sum_i sum_k [ a*u_ik^m + b*t_ik^eta ] * d2_ik
    """
    W = a * (U ** m) + b * (T ** eta)
    return float(np.sum(W * D2))


def pfcm(
        X: np.ndarray,
        k: int,
        m: float = 2.0,
        eta: float = 2.0,
        a: float = 1.0,
        b: float = 1.0,
        max_iter: int = 300,
        tol: float = 1e-5,
        seed: int = 0,
) -> PFCMResult:
    """
    Run PFCM on X (n,d).
    """
    rng = np.random.default_rng(seed)
    n, _ = X.shape

    init_idx = rng.choice(n, size=k, replace=False)
    V = X[init_idx].copy()

    D2 = _squared_euclidean(X, V)
    U = _update_U_from_distances(D2, m=m)

    um = U ** m
    gamma0 = np.sum(um * D2, axis=0) / (np.sum(um, axis=0) + EPS)
    gamma0 = np.maximum(gamma0, EPS)
    T = _update_T_from_distances(D2, gamma=gamma0, eta=eta, b=b)
    gamma = gamma0

    prev_obj: Optional[float] = None
    for it in range(1, max_iter + 1):
        V_new = _update_centers(X, U, T, a=a, b=b, m=m, eta=eta)
        D2 = _squared_euclidean(X, V_new)

        U = _update_U_from_distances(D2, m=m)
        gamma = _update_gamma(D2, U, T, a=a, b=b, m=m, eta=eta)
        T = _update_T_from_distances(D2, gamma=gamma, eta=eta, b=b)

        obj = _objective(D2, U, T, a=a, b=b, m=m, eta=eta)
        center_shift = float(np.max(np.linalg.norm(V_new - V, axis=1)))
        V = V_new

        if prev_obj is not None:
            rel_impr = abs(prev_obj - obj) / (abs(prev_obj) + EPS)
            if (center_shift < tol) or (rel_impr < tol):
                return PFCMResult(
                    centers=V,
                    U=U,
                    T=T,
                    gamma=gamma,
                    objective=obj,
                    n_iter=it,
                )

        prev_obj = obj

    return PFCMResult(
        centers=V,
        U=U,
        T=T,
        gamma=gamma,
        objective=float(prev_obj) if prev_obj is not None else float("nan"),
        n_iter=max_iter,
    )


def xie_beni_index(X: np.ndarray, centers: np.ndarray, U: np.ndarray, m: float) -> float:
    """
    Xie-Beni index (lower is better):
      XB = sum_i sum_k u_ik^m * ||x_i - v_k||^2 / (n * min_{p!=q} ||v_p - v_q||^2)
    """
    D2 = _squared_euclidean(X, centers)
    num = float(np.sum((U ** m) * D2))
    n = X.shape[0]

    k = centers.shape[0]
    min_sep = float("inf")
    for i in range(k):
        for j in range(i + 1, k):
            sep = float(np.sum((centers[i] - centers[j]) ** 2))
            if sep < min_sep:
                min_sep = sep
    min_sep = max(min_sep, EPS)
    return num / (n * min_sep)


def partition_coefficient(U: np.ndarray) -> float:
    return float(np.mean(np.sum(U * U, axis=1)))


def classification_entropy(U: np.ndarray) -> float:
    return float(-np.mean(np.sum(U * np.log(U + EPS), axis=1)))


def prototype_silhouette(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """
    Prototype-based silhouette using distances to centers:
      a_i = dist(x_i, center_{label_i})
      b_i = min_{k != label_i} dist(x_i, center_k)
      s_i = (b_i - a_i) / max(a_i, b_i)
    """
    D2 = _squared_euclidean(X, centers)
    D = np.sqrt(np.maximum(D2, 0.0))

    n, _ = D.shape
    a = D[np.arange(n), labels]

    D_other = D.copy()
    D_other[np.arange(n), labels] = np.inf
    b = np.min(D_other, axis=1)

    s = (b - a) / (np.maximum(a, b) + EPS)
    return float(np.mean(s))


def fuzzy_prototype_silhouette(X: np.ndarray, centers: np.ndarray, U: np.ndarray, m: float) -> float:
    """
    FSI implemented as a fuzzy-weighted prototype silhouette:
    - compute prototype silhouette s_i from argmax(U) assignment
    - weight each s_i by w_i = (max_k u_ik)^m
    """
    labels = np.argmax(U, axis=1)
    D2 = _squared_euclidean(X, centers)
    D = np.sqrt(np.maximum(D2, 0.0))

    n = X.shape[0]
    a = D[np.arange(n), labels]

    D_other = D.copy()
    D_other[np.arange(n), labels] = np.inf
    b = np.min(D_other, axis=1)

    s = (b - a) / (np.maximum(a, b) + EPS)
    w = np.max(U, axis=1) ** m
    return float(np.sum(w * s) / (np.sum(w) + EPS))


def cluster_stability_ari(
        X: np.ndarray,
        k: int,
        *,
        m: float,
        eta: float,
        a: float,
        b: float,
        max_iter: int,
        tol: float,
        seed: int,
        n_runs: int = 8,
) -> float:
    """
    Stability via mean pairwise ARI across multiple runs (different seeds).
    """
    labels_list: List[np.ndarray] = []
    for r in range(n_runs):
        res = pfcm(
            X,
            k=k,
            m=m,
            eta=eta,
            a=a,
            b=b,
            max_iter=max_iter,
            tol=tol,
            seed=seed + 999 * r + 12345 * k,
        )
        labels_list.append(np.argmax(res.U, axis=1))

    aris: List[float] = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))

    return float(np.mean(aris)) if aris else float("nan")


def interpretability_score(U: np.ndarray, T: np.ndarray) -> float:
    """
    Simple interpretability proxy:
      0.5*E[max_k u_ik] + 0.5*E[max_k t_ik]
    """
    return float(0.5 * np.mean(np.max(U, axis=1)) + 0.5 * np.mean(np.max(T, axis=1)))


def make_pca_scatter_plot(
        X: np.ndarray,
        actor_ids: np.ndarray,
        U: np.ndarray,
        T: np.ndarray,
        centers: np.ndarray,
        outpath: str,
        title: str,
        seed: int = 0,
) -> None:
    """
    PCA projection to 2D:
    - color = argmax membership
    - size  = max typicality
    - centers as X markers
    - highlight one random user as red point
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    pca = PCA(n_components=2, random_state=seed)
    X2 = pca.fit_transform(X)
    C2 = pca.transform(centers)

    labels = np.argmax(U, axis=1)
    max_t = np.max(T, axis=1)
    idx_star = int(rng.integers(0, n))

    plt.figure(figsize=(10, 8))
    plt.scatter(
        X2[:, 0],
        X2[:, 1],
        c=labels,
        s=10 + 60 * max_t,
        alpha=0.75
    )
    plt.scatter(C2[:, 0], C2[:, 1], marker="X", s=250, edgecolor="k")
    plt.scatter(X2[idx_star, 0], X2[idx_star, 1], c="red", s=120, marker="o", edgecolor="k")

    plt.title(title + f"\nHighlighted actor_id: {actor_ids[idx_star]}")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def make_membership_heatmap(U: np.ndarray, outpath: str, title: str) -> None:
    order = np.argsort(-np.max(U, axis=1))
    U_sorted = U[order]

    plt.figure(figsize=(10, 8))
    plt.imshow(U_sorted, aspect="auto")
    plt.colorbar(label="membership u")
    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel("Users (sorted by max u)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def make_validity_curves(metrics_df: pd.DataFrame, outdir: str) -> None:
    ks = metrics_df["k"].to_numpy()

    def _plot(col: str, ylabel: str, filename: str):
        plt.figure(figsize=(10, 6))
        plt.plot(ks, metrics_df[col].to_numpy(), marker="o")
        plt.xlabel("k (clusters)")
        plt.ylabel(ylabel)
        plt.title(f"PFCM validity curve: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=180)
        plt.close()

    _plot("xb", "Xie-Beni (lower better)", "validity_xb.png")
    _plot("pc", "Partition Coefficient (higher better)", "validity_pc.png")
    _plot("ce", "Classification Entropy (lower better)", "validity_ce.png")
    _plot("silhouette", "Silhouette (higher better)", "validity_silhouette.png")
    _plot("fsi_fuzzy_proto_silhouette", "FSI (higher better)", "validity_fsi.png")
    _plot("cluster_stability_ari", "Stability (mean ARI; higher better)", "validity_stability_ari.png")
    _plot("interpretability", "Interpretability (higher better)", "validity_interpretability.png")


def preprocess_input(X_raw: np.ndarray, input_mode: str) -> tuple[np.ndarray, str]:
    """
    Preprocess input according to mode.
    """
    if np.isnan(X_raw).any() or np.isinf(X_raw).any():
        raise ValueError("Input contains NaN or infinite values before preprocessing.")

    min_val = float(np.nanmin(X_raw))

    if input_mode == "raw_counts":
        if min_val <= -1.0:
            raise ValueError(
                f"input_mode='raw_counts' but minimum value is {min_val:.6f}, so log1p is invalid."
            )
        X = np.log1p(X_raw)
        X = StandardScaler().fit_transform(X)
        transform_name = "log1p + StandardScaler"

    elif input_mode == "preprocessed":
        X = X_raw.copy()
        transform_name = "preprocessed input (used as-is)"

    else:  # auto
        if min_val <= -1.0:
            X = X_raw.copy()
            transform_name = "auto-detected preprocessed input (used as-is)"
        elif min_val < 0.0:
            X = StandardScaler().fit_transform(X_raw)
            transform_name = "auto-detected mixed-sign input -> StandardScaler only"
        else:
            X = np.log1p(X_raw)
            X = StandardScaler().fit_transform(X)
            transform_name = "auto-detected raw counts -> log1p + StandardScaler"

    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input contains NaN or infinite values after preprocessing.")

    return X, transform_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="/mnt/data/leonardo_activity_unpacked.csv")
    ap.add_argument("--outdir", type=str, default="PFCM_output_1_12")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=12)

    ap.add_argument("--m", type=float, default=2.0, help="FCM fuzzifier (>1)")
    ap.add_argument("--eta", type=float, default=2.0, help="PCM typicality exponent (>1)")
    ap.add_argument("--a", type=float, default=1.0, help="weight for FCM term")
    ap.add_argument("--b", type=float, default=1.0, help="weight for PCM term")

    ap.add_argument("--max_iter", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--n_init", type=int, default=3, help="random restarts per k (best objective kept)")
    ap.add_argument(
        "--stability_runs",
        type=int,
        default=0,
        help="runs for stability ARI per k (0 => max(5, 2*n_init))"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--input_mode",
        type=str,
        default="auto",
        choices=["auto", "raw_counts", "preprocessed"],
        help="auto: infer from data; raw_counts: apply log1p+scaler; preprocessed: use numeric values as-is"
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    if "actor_id" not in df.columns:
        raise ValueError("Expected an 'actor_id' column.")

    actor_ids = df["actor_id"].astype(str).to_numpy()

    num_cols = [c for c in df.columns if c != "actor_id" and pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) == 0:
        raise ValueError("No numeric feature columns found besides actor_id.")

    X_raw = df[num_cols].to_numpy(dtype=float)
    X, transform_name = preprocess_input(X_raw, args.input_mode)

    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump(
            {
                "input": args.input,
                "features": num_cols,
                "k_range": [args.kmin, args.kmax],
                "m": args.m,
                "eta": args.eta,
                "a": args.a,
                "b": args.b,
                "max_iter": args.max_iter,
                "tol": args.tol,
                "n_init": args.n_init,
                "stability_runs": args.stability_runs,
                "seed": args.seed,
                "input_mode": args.input_mode,
                "transform": transform_name,
            },
            f,
            indent=2
        )

    metrics_rows = []
    best_by_xb = {"k": None, "xb": float("inf")}

    for k in range(args.kmin, args.kmax + 1):
        k_dir = os.path.join(args.outdir, f"k_{k:02d}")
        os.makedirs(k_dir, exist_ok=True)

        # multiple restarts -> choose best objective
        best_res: Optional[PFCMResult] = None
        best_obj = float("inf")
        failed_runs = 0

        for r in range(args.n_init):
            seed = args.seed + 1000 * k + r
            try:
                res = pfcm(
                    X,
                    k=k,
                    m=args.m,
                    eta=args.eta,
                    a=args.a,
                    b=args.b,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    seed=seed
                )

                if (
                        np.isnan(res.objective)
                        or np.isnan(res.centers).any()
                        or np.isnan(res.U).any()
                        or np.isnan(res.T).any()
                ):
                    failed_runs += 1
                    continue

                if res.objective < best_obj:
                    best_obj = res.objective
                    best_res = res

            except Exception as e:
                print(f"[k={k}] restart {r} failed: {e}")
                failed_runs += 1
                continue

        if best_res is None:
            raise RuntimeError(
                f"All {args.n_init} PFCM restarts failed for k={k}. "
                f"Check preprocessing and input values."
            )

        xb = xie_beni_index(X, best_res.centers, best_res.U, m=args.m)
        pc = partition_coefficient(best_res.U)
        ce = classification_entropy(best_res.U)

        hard_labels = np.argmax(best_res.U, axis=1)

        try:
            sil = float(silhouette_score(X, hard_labels, metric="euclidean"))
        except Exception:
            sil = float("nan")

        proto_sil = prototype_silhouette(X, best_res.centers, hard_labels)
        fsi = fuzzy_prototype_silhouette(X, best_res.centers, best_res.U, m=args.m)

        n_runs = args.stability_runs if args.stability_runs > 0 else max(5, args.n_init * 2)
        stability = cluster_stability_ari(
            X,
            k,
            m=args.m,
            eta=args.eta,
            a=args.a,
            b=args.b,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=args.seed,
            n_runs=n_runs
        )

        interp = interpretability_score(best_res.U, best_res.T)

        row = {
            "k": k,
            "objective": best_res.objective,
            "xb": xb,
            "pc": pc,
            "ce": ce,
            "silhouette": sil,
            "proto_silhouette": proto_sil,
            "fsi_fuzzy_proto_silhouette": fsi,
            "cluster_stability_ari": stability,
            "interpretability": interp,
            "n_iter": best_res.n_iter,
            "failed_restarts": failed_runs,
        }
        metrics_rows.append(row)

        if xb < best_by_xb["xb"]:
            best_by_xb = {"k": k, "xb": xb}

        # Save memberships/typicalities
        out_df = pd.DataFrame({"actor_id": actor_ids})
        for j in range(k):
            out_df[f"u_{j+1}"] = best_res.U[:, j]
        for j in range(k):
            out_df[f"t_{j+1}"] = best_res.T[:, j]
        out_df["cluster_argmax_u"] = np.argmax(best_res.U, axis=1) + 1
        out_df["max_u"] = np.max(best_res.U, axis=1)
        out_df["max_t"] = np.max(best_res.T, axis=1)
        out_df.to_csv(os.path.join(k_dir, "memberships_typicalities.csv"), index=False)

        # Save centers
        centers_df = pd.DataFrame(best_res.centers, columns=num_cols)
        centers_df.insert(0, "cluster", np.arange(1, k + 1))
        centers_df.to_csv(os.path.join(k_dir, "centers.csv"), index=False)

        # Per-k metrics JSON
        per_k_json = dict(row)
        per_k_json["gamma"] = best_res.gamma.tolist()
        per_k_json["stability_runs"] = int(n_runs)
        with open(os.path.join(k_dir, f"metrics_k{k:02d}.json"), "w") as f:
            json.dump(per_k_json, f, indent=2)

        # Plots
        make_pca_scatter_plot(
            X=X,
            actor_ids=actor_ids,
            U=best_res.U,
            T=best_res.T,
            centers=best_res.centers,
            outpath=os.path.join(k_dir, f"pca_scatter_k{k:02d}.png"),
            title=f"PFCM (k={k}) PCA scatter: color=argmax(U), size=max(T)",
            seed=args.seed + k
        )
        make_membership_heatmap(
            U=best_res.U,
            outpath=os.path.join(k_dir, f"membership_heatmap_k{k:02d}.png"),
            title=f"PFCM (k={k}) membership heatmap (sorted by max u)"
        )

        print(
            f"[k={k}] obj={best_res.objective:.3e} xb={xb:.3e} "
            f"sil={sil:.3f} fsi={fsi:.3f} stabARI={stability:.3f} "
            f"interp={interp:.3f} iters={best_res.n_iter} failed_restarts={failed_runs}"
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("k")
    metrics_df.to_csv(os.path.join(args.outdir, "pfcm_summary_metrics.csv"), index=False)

    # Compact summary CSV
    cluster_quality_summary = metrics_df[
        ["k", "silhouette", "fsi_fuzzy_proto_silhouette", "cluster_stability_ari", "interpretability"]
    ].copy()
    cluster_quality_summary.rename(
        columns={"fsi_fuzzy_proto_silhouette": "fsi"},
        inplace=True
    )
    cluster_quality_summary.to_csv(
        os.path.join(args.outdir, "pfcm_cluster_quality_summary.csv"),
        index=False
    )

    make_validity_curves(metrics_df, args.outdir)

    best_k = best_by_xb["k"]
    with open(os.path.join(args.outdir, "best_model_by_xb.txt"), "w") as f:
        f.write(f"Best k by Xie-Beni (lower better): k={best_k}, xb={best_by_xb['xb']:.6e}\n")

    summary_json = {
        "kmin": args.kmin,
        "kmax": args.kmax,
        "best_k_by_xb": int(best_k) if best_k is not None else None,
        "transform": transform_name,
        "metrics": metrics_df.to_dict(orient="records"),
    }
    with open(os.path.join(args.outdir, "pfcm_summary_metrics.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\nBest k by Xie-Beni: k={best_k}, xb={best_by_xb['xb']:.3e}")
    print(f"Outputs written to: {args.outdir}")
    print(f"Compact summary CSV: {os.path.join(args.outdir, 'pfcm_cluster_quality_summary.csv')}")


if __name__ == "__main__":
    main()