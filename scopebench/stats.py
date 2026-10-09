"""Statistics (plan §9.1–9.2): patient-cluster bootstrap, paired contrasts, Holm correction,
Fisher-z intervals for correlations, mutual information. Loads no model. Deterministic given the seed."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


def cluster_bootstrap_mean(values: Sequence[float], clusters: Sequence[str], n_boot: int = 10000, seed: int = 2027) -> tuple[float, float, float]:
    """Mean of `values` with a 95 % interval from resampling clusters (patients) with replacement."""
    by = defaultdict(list)
    for v, c in zip(values, clusters):
        by[c].append(float(v))
    keys = list(by)
    sums = np.array([sum(by[k]) for k in keys])
    counts = np.array([len(by[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    point = float(sums.sum() / counts.sum())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_cluster_contrast(a: dict[str, float], b: dict[str, float], task_cluster: dict[str, str], n_boot: int = 10000, seed: int = 2027) -> dict:
    """Δ = mean(b) − mean(a) over the common task ids, patient-cluster bootstrap interval and a two-sided
    bootstrap p-value (fraction of resamples with the opposite sign, doubled, floored at 1/n_boot)."""
    ids = sorted(set(a) & set(b))
    diffs = np.array([b[i] - a[i] for i in ids], dtype=float)
    clusters = [task_cluster[i] for i in ids]
    by = defaultdict(list)
    for d, c in zip(diffs, clusters):
        by[c].append(d)
    keys = list(by)
    sums = np.array([sum(by[k]) for k in keys])
    counts = np.array([len(by[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    delta = float(diffs.mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    if delta >= 0:
        p = 2 * float((boot <= 0).mean())
    else:
        p = 2 * float((boot >= 0).mean())
    p = max(min(p, 1.0), 1.0 / n_boot)
    return {"n": len(ids), "n_clusters": len(keys), "delta": delta, "ci_low": float(lo), "ci_high": float(hi), "p_bootstrap": p}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        val = min(1.0, (m - rank) * p)
        running = max(running, val)
        adjusted[name] = running
    return adjusted


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n < 4 or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    return (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    return float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])


def mutual_information_bits(x: Iterable[int], y: Iterable[int]) -> float:
    """MI between two binary/categorical variables, in bits."""
    x = list(x); y = list(y)
    n = len(x)
    joint = defaultdict(int); px = defaultdict(int); py = defaultdict(int)
    for a, b in zip(x, y):
        joint[(a, b)] += 1; px[a] += 1; py[b] += 1
    mi = 0.0
    for (a, b), c in joint.items():
        pab = c / n
        mi += pab * math.log2(pab / ((px[a] / n) * (py[b] / n)))
    return mi


def contingency(x: Iterable[int], y: Iterable[int]) -> dict:
    x = list(x); y = list(y)
    tp = sum(1 for a, b in zip(x, y) if a and b); tn = sum(1 for a, b in zip(x, y) if not a and not b)
    fp = sum(1 for a, b in zip(x, y) if not a and b); fn = sum(1 for a, b in zip(x, y) if a and not b)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": len(x), "mi_bits": mutual_information_bits(x, y)}
