"""Within-subject reactivity + noise-floor calibration over extracted markers.

The question this answers: when a sound condition is compared to that same patient's
silence recording, is the marker change LARGER than the change we would see between two
stretches of the same silence recording?

Terms:
  reactivity  = per-subject (condition mean) - (reference mean), reference default "No Music".
  noise floor = the spread of split-half differences WITHIN each reference recording,
                i.e. the change we measure when nothing changed.

Ratio-like markers (ADR, absolute band powers) are log10-transformed before differencing, so
a delta is a fold-change and the noise distribution is roughly symmetric. Relative powers and
entropies are already bounded and are differenced as-is.

Scale note (matters for interpretation): a split-half difference is computed from two halves of
one recording, so each half carries ~half the epochs of a full recording. Its sampling variance
is therefore ~2x that of a difference between two full recordings, making the raw floor
CONSERVATIVE by ~sqrt(2). Both the raw floor and the sqrt(2)-matched floor are reported.

Bigger caveat, stated everywhere it is relevant: a split-half floor lives inside a single
recording, so it captures epoch-to-epoch variance only. Reactivity compares two SEPARATE
recordings, which additionally carries session drift (electrode impedance, sedation, arousal
state). The true floor for a between-recording comparison is therefore >= this one. Treat these
numbers as a lower bound on the floor and an upper bound on detectability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REFERENCE = "No Music"
COND_ORDER = ["No Music", "Pink Noise", "Recorded Music", "Live Music"]

# Markers differenced in log10 space (strictly positive, ratio-like).
LOG_MARKERS = {"adr", "delta_abs", "theta_abs", "alpha_abs", "beta_abs"}


def uses_log(marker: str) -> bool:
    return marker in LOG_MARKERS


def _values(df: pd.DataFrame, marker: str, log: bool) -> pd.Series:
    v = pd.to_numeric(df[marker], errors="coerce")
    if log:
        v = v.where(v > 0)
        v = np.log10(v)
    return v


def load_region_epochs(path: str) -> pd.DataFrame:
    """Read the per-epoch marker table emitted by scripts/extract_all_markers.py."""
    return pd.read_csv(path)


def select(df: pd.DataFrame, marker: str, region: str = "whole_scalp") -> pd.DataFrame:
    """Narrow the epoch table to one marker in one region, dropping unusable rows."""
    if region not in set(df["region"]):
        raise ValueError(f"unknown region {region!r}")
    if marker not in df.columns:
        raise ValueError(f"unknown marker {marker!r}")
    log = uses_log(marker)
    out = df.loc[df["region"] == region, ["subject", "condition", "epoch"]].copy()
    out["value"] = _values(df.loc[df["region"] == region], marker, log).to_numpy()
    return out.dropna(subset=["value"]).reset_index(drop=True)


def subject_condition_means(sel: pd.DataFrame) -> pd.DataFrame:
    """Collapse epochs to one value per subject x condition."""
    g = sel.groupby(["subject", "condition"])["value"]
    return pd.DataFrame({"value": g.mean(), "sd": g.std(ddof=1), "n_epochs": g.size()}).reset_index()


def noise_floor(sel: pd.DataFrame, reference: str = REFERENCE,
                scheme: str = "halves") -> pd.DataFrame:
    """Per-subject split-half difference within the reference recording.

    scheme "halves": first half vs second half (sensitive to drift within the recording).
    scheme "oddeven": alternating epochs (drift-free, so a purer measurement-noise floor).
    """
    ref = sel[sel["condition"] == reference]
    rows = []
    for subject, g in ref.groupby("subject"):
        g = g.sort_values("epoch")
        v = g["value"].to_numpy()
        if len(v) < 4:
            continue
        if scheme == "halves":
            mid = len(v) // 2
            a, b = v[:mid], v[mid:]
        elif scheme == "oddeven":
            a, b = v[0::2], v[1::2]
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        rows.append({"subject": subject, "half_a": a.mean(), "half_b": b.mean(),
                     "floor_diff": b.mean() - a.mean(), "n_epochs": len(v)})
    return pd.DataFrame(rows)


def floor_summary(floors: pd.DataFrame) -> dict:
    """Turn the per-subject split-half differences into floor thresholds."""
    d = floors["floor_diff"].to_numpy()
    if len(d) == 0:
        return {"n": 0}
    absd = np.abs(d)
    raw95 = float(np.percentile(absd, 95))
    return {
        "n": int(len(d)),
        "signed_mean": float(d.mean()),
        "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        "abs_median": float(np.median(absd)),
        "abs_p95": raw95,
        # sqrt(2) correction: halves hold ~half the epochs of a full recording.
        "abs_p95_matched": raw95 / np.sqrt(2.0),
        "scheme_note": "lower bound; captures epoch variance only, not between-recording drift",
    }


def reactivity(means: pd.DataFrame, reference: str = REFERENCE) -> pd.DataFrame:
    """Per-subject condition-minus-reference deltas. Subjects lacking the reference are dropped."""
    ref = means[means["condition"] == reference].set_index("subject")["value"]
    out = means[means["condition"] != reference].copy()
    out["reference_value"] = out["subject"].map(ref)
    out = out.dropna(subset=["reference_value"])
    out["delta"] = out["value"] - out["reference_value"]
    return out.reset_index(drop=True)


def group_stats(delta: pd.Series) -> dict:
    """Paired-difference stats: t-test, Wilcoxon, Cohen's dz, 95% CI of the mean."""
    from scipy import stats
    d = pd.to_numeric(delta, errors="coerce").dropna().to_numpy()
    n = len(d)
    if n < 2:
        return {"n": n}
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    sem = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t, p_t = stats.ttest_1samp(d, 0.0)
    try:
        _, p_w = stats.wilcoxon(d)
    except ValueError:                      # all-zero differences
        p_w = float("nan")
    return {"n": n, "mean": mean, "sd": sd,
            "ci_lo": mean - tcrit * sem, "ci_hi": mean + tcrit * sem,
            "t": float(t), "p_t": float(p_t), "p_wilcoxon": float(p_w),
            "cohen_dz": mean / sd if sd > 0 else float("nan")}


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full(p.shape, np.nan)
    q = p[ok]
    if q.size == 0:
        return out
    order = np.argsort(q)
    ranked = q[order] * q.size / (np.arange(q.size) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(ranked)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def split_half_reliability(half_a, half_b) -> dict:
    """Reliability of a per-subject index from two independent halves of that subject's data.

    Spearman-Brown steps the half-length correlation up to the reliability of the full-length
    index. A reactivity index needs decent reliability before any correlation with a clinical
    score is worth attempting; a near-zero value means the index is mostly noise.
    """
    from scipy import stats
    a = np.asarray(half_a, dtype=float)
    b = np.asarray(half_b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return {"n": int(len(a))}
    r = float(stats.pearsonr(a, b).statistic)
    rho = float(stats.spearmanr(a, b).statistic)
    return {"n": int(len(a)), "pearson_r": r, "spearman_rho": rho,
            "spearman_brown": 2 * r / (1 + r) if r > -1 else float("nan")}


def group_contrast(x, y) -> dict:
    """Between-group comparison: Mann-Whitney, AUC, Cliff's delta, Hedges' g.

    AUC is the probability a random member of x exceeds a random member of y, so 0.5 is no
    separation. Reported alongside p because with n ~ 86 per group a small p can still mean a
    clinically useless overlap.
    """
    from scipy import stats
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy()
    if len(x) < 3 or len(y) < 3:
        return {"n_x": int(len(x)), "n_y": int(len(y))}
    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    auc = float(u) / (len(x) * len(y))
    sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1))
                 / (len(x) + len(y) - 2))
    return {"n_x": int(len(x)), "n_y": int(len(y)),
            "mean_x": float(x.mean()), "mean_y": float(y.mean()),
            "median_x": float(np.median(x)), "median_y": float(np.median(y)),
            "p_mannwhitney": float(p), "auc": auc, "cliffs_delta": 2 * auc - 1,
            "hedges_g": float((x.mean() - y.mean()) / sp) if sp > 0 else float("nan")}


def per_subject_epoch_tests(sel: pd.DataFrame, reference: str = REFERENCE) -> pd.DataFrame:
    """Welch t-test on epoch values, condition vs reference, within each subject.

    Anticonservative by construction: consecutive epochs of one recording are correlated and
    are not independent samples, so these p-values run small. Kept as a per-subject effect-size
    readout (hedges_g), not as the primary inference.
    """
    from scipy import stats
    rows = []
    for subject, g in sel.groupby("subject"):
        ref = g.loc[g["condition"] == reference, "value"].to_numpy()
        if len(ref) < 4:
            continue
        for cond, gc in g[g["condition"] != reference].groupby("condition"):
            x = gc["value"].to_numpy()
            if len(x) < 4:
                continue
            t, p = stats.ttest_ind(x, ref, equal_var=False)
            n1, n2 = len(x), len(ref)
            sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * ref.var(ddof=1)) / (n1 + n2 - 2))
            rows.append({"subject": subject, "condition": cond,
                         "delta": float(x.mean() - ref.mean()),
                         "t": float(t), "p_epochwise": float(p),
                         "hedges_g": float((x.mean() - ref.mean()) / sp) if sp > 0 else np.nan})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_epochwise"] = bh_fdr(out["p_epochwise"])
    return out


def calibrate(df: pd.DataFrame, marker: str = "adr", region: str = "whole_scalp",
              reference: str = REFERENCE, scheme: str = "halves") -> dict:
    """Full calibration for one marker in one region.

    Returns per-subject reactivity flagged against the floor, group stats per condition,
    and the floor summary.
    """
    sel = select(df, marker, region)
    means = subject_condition_means(sel)
    react = reactivity(means, reference)
    floors = noise_floor(sel, reference, scheme)
    fs = floor_summary(floors)

    react = react.merge(floors[["subject", "floor_diff"]], on="subject", how="left")
    react["abs_delta"] = react["delta"].abs()
    if fs.get("n"):
        react["clears_group_floor"] = react["abs_delta"] > fs["abs_p95"]
        react["clears_matched_floor"] = react["abs_delta"] > fs["abs_p95_matched"]
        react["clears_own_floor"] = react["abs_delta"] > react["floor_diff"].abs()
    react["log_space"] = uses_log(marker)

    groups = []
    for cond in [c for c in COND_ORDER if c != reference]:
        d = react.loc[react["condition"] == cond, "delta"]
        if not len(d):
            continue
        row = {"condition": cond, **group_stats(d)}
        sub = react[react["condition"] == cond]
        for col in ("clears_group_floor", "clears_matched_floor", "clears_own_floor"):
            if col in sub:
                row[f"n_{col}"] = int(sub[col].sum())
        groups.append(row)
    group = pd.DataFrame(groups)
    if len(group) and "p_wilcoxon" in group:
        group["q_wilcoxon"] = bh_fdr(group["p_wilcoxon"])

    return {"marker": marker, "region": region, "reference": reference, "scheme": scheme,
            "log_space": uses_log(marker), "means": means, "reactivity": react,
            "floors": floors, "floor": fs, "group": group}
