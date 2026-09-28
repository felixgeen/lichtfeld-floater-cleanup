"""Outlier removal core for Gaussian splats (pure numpy/scipy, thread-safe).

Two methods:

global  Classic Statistical Outlier Removal. Each point's mean distance to its
        K nearest neighbours is compared with ONE threshold for the whole
        point set: mean + std_ratio * std. Good for floaters out in empty
        space. Weak near dense geometry: in a big outdoor scene the threshold
        is set by the sparse background, so a floater a few cm from a cable
        never exceeds it.

local   Local density ratio (LOF-style). Each point's mean kNN distance is
        divided by the median of its neighbours' mean kNN distances. A
        floater hovering next to a cable is much sparser than the cable
        Gaussians it sits beside, so its ratio is high even though its
        absolute distances are tiny. Scale-free, so it works the same on
        fine detail and on the background.
"""

import numpy as np
from scipy.spatial import cKDTree

QUERY_CHUNK = 262_144  # points per KD-tree query batch (keeps RAM bounded)


def _build_tree(points):
    return cKDTree(points, leafsize=32, balanced_tree=False, compact_nodes=False)


def _mean_knn(tree, points, kq, progress, cancel):
    n = len(points)
    md = np.empty(n, dtype=np.float32)
    for start in range(0, n, QUERY_CHUNK):
        if cancel is not None and cancel():
            return None
        end = min(start + QUERY_CHUNK, n)
        dist, _ = tree.query(points[start:end], k=kq, workers=-1)
        md[start:end] = dist[:, 1:].mean(axis=1)
        if progress is not None:
            progress(end / n)
    return md


def _local_ratio(tree, points, md, kq, progress, cancel):
    n = len(points)
    ratio = np.empty(n, dtype=np.float32)
    floor = max(float(np.median(md)) * 1e-3, 1e-12)
    for start in range(0, n, QUERY_CHUNK):
        if cancel is not None and cancel():
            return None
        end = min(start + QUERY_CHUNK, n)
        _, idx = tree.query(points[start:end], k=kq, workers=-1)
        ref = np.median(md[idx[:, 1:]], axis=1)
        ratio[start:end] = md[start:end] / np.maximum(ref, floor)
        if progress is not None:
            progress(end / n)
    return ratio


def outlier_mask(points, k=20, method="global", std_ratio=2.0, local_ratio=2.0,
                 passes=1, progress=None, cancel=None):
    """Flag outliers in an (N, 3) array.

    Returns (mask, stats); mask is bool (True = outlier), or None if cancelled.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    n = len(points)
    outlier = np.zeros(n, dtype=bool)
    stats = []
    all_idx = np.arange(n)
    threshold = None  # global method: fixed after pass 1 so later passes can't tighten it

    for p in range(passes):
        live = all_idx[~outlier]
        if len(live) <= k + 1:
            break
        pts = points[live]
        kq = min(k + 1, len(pts))
        steps = 1 if method == "global" else 2

        def sub(step, p=p):
            def f(x):
                if progress is not None:
                    progress((p + (step + x) / steps) / passes)
            return f

        tree = _build_tree(pts)
        md = _mean_knn(tree, pts, kq, sub(0), cancel)
        if md is None:
            return None, stats

        if method == "local":
            score = _local_ratio(tree, pts, md, kq, sub(1), cancel)
            if score is None:
                return None, stats
            bad = score > local_ratio
            stats.append({"pass": p + 1, "method": "local",
                          "median_score": float(np.median(score)),
                          "p99_score": float(np.percentile(score, 99)),
                          "threshold": float(local_ratio), "removed": int(bad.sum())})
        else:
            mu, sd = float(md.mean()), float(md.std())
            if threshold is None:
                threshold = mu + std_ratio * sd
            bad = md > threshold
            stats.append({"pass": p + 1, "method": "global", "mean_dist": mu,
                          "std_dist": sd, "threshold": threshold,
                          "removed": int(bad.sum())})

        outlier[live[bad]] = True
        if not bad.any():
            break

    if progress is not None:
        progress(1.0)
    return outlier, stats


def build_removal_mask(means, alive, opacity, k, method, std_ratio, local_ratio, passes,
                       min_opacity=None, region=None, progress=None, cancel=None):
    """Full per-node pipeline: non-finite check, optional opacity prune, outlier test.

    means:   (N, 3) positions
    alive:   (N,) bool, False for Gaussians already soft-deleted (left untouched)
    opacity: (N,) activated opacity in [0, 1], or None to skip opacity pruning
    region:  (N,) bool or None. If given, only these Gaussians are analysed and
             can be removed, and the statistics come from this region alone.

    Returns a dict with the removal mask and counts, or None if cancelled.
    """
    n = len(means)
    remove = np.zeros(n, dtype=bool)

    scope = alive.copy()
    if region is not None:
        scope &= region

    finite = np.isfinite(means).all(axis=1)
    nonfinite = scope & ~finite
    remove |= nonfinite
    candidates = scope & finite

    n_opacity = 0
    if opacity is not None and min_opacity is not None:
        low = candidates & (opacity < min_opacity)
        n_opacity = int(low.sum())
        remove |= low
        candidates &= ~low

    idx = np.flatnonzero(candidates)
    n_outlier = 0
    stats = []
    if len(idx) > k + 1:
        out, stats = outlier_mask(means[idx], k, method, std_ratio, local_ratio, passes,
                                  progress=progress, cancel=cancel)
        if out is None:
            return None
        remove[idx[out]] = True
        n_outlier = int(out.sum())

    return {
        "mask": remove,
        "n_total": n,
        "n_alive": int(scope.sum()),
        "n_sor": n_outlier,
        "n_opacity": n_opacity,
        "n_nonfinite": int(nonfinite.sum()),
        "stats": stats,
    }
