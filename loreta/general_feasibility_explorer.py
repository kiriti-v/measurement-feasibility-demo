"""Dataset-neutral repeated-measures marker feasibility explorer.

The analysis contract is one row per subject x condition x epoch (optionally x region and block),
with one or more numeric marker columns. Built-in adapters expose the existing ICU music and
OpenNeuro eyes-protocol results through that contract; a staging workspace maps ordinary tabular
uploads onto the same contract without changing the analysis code.

Run from the repository root:
    python -m loreta.general_feasibility_explorer
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update

from loreta import reactivity as rx
from loreta.staging_recipes import make_recipe, parse_recipe, recipe_json, validate_recipe


MUSIC_PATH = os.environ.get(
    "FEASIBILITY_MUSIC_PATH", os.path.join("results", "markers", "region_epochs.csv"))
OPENNEURO_PATH = os.environ.get(
    "FEASIBILITY_OPENNEURO_PATH", os.path.join("results", "openneuro", "block_epochs.csv"))
EPOCH_SECONDS = 3.0
Z_975 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143

MARKER_LABELS = {
    "adr": "alpha/delta ratio", "delta_abs": "absolute delta",
    "delta_rel": "relative delta", "theta_abs": "absolute theta",
    "theta_rel": "relative theta", "alpha_abs": "absolute alpha",
    "alpha_rel": "relative alpha", "beta_abs": "absolute beta",
    "beta_rel": "relative beta", "permen": "permutation entropy",
    "sampen": "sample entropy", "apen": "approximate entropy",
}
MARKER_ORDER = ["adr", "alpha_rel", "delta_rel", "theta_rel", "beta_rel", "permen",
                "alpha_abs", "delta_abs", "theta_abs", "beta_abs", "sampen", "apen"]
REGION_ORDER = ["whole_scalp", "temporal", "frontal", "central", "parietal",
                "occipital", "left", "right", "midline"]
META_COLUMNS = {
    "subject", "condition", "epoch", "region", "block", "state", "t_start",
    "n_channels", "group", "is_baseline",
}

PAPER = "#f6f8f9"
WHITE = "#ffffff"
INK = "#17212b"
MUTED = "#60717c"
TRACE = "#1d5d78"
TRACE_LIGHT = "#92b5c2"
MIST = "#e7eef1"
AMBER = "#c6821e"


@dataclass
class DatasetBundle:
    key: str
    label: str
    epochs: pd.DataFrame
    epoch_seconds: float = EPOCH_SECONDS
    block_col: str | None = None
    floor_factor: float = 1 / np.sqrt(2.0)
    default_condition: str | None = None
    default_reference: str | None = None
    default_marker: str | None = None
    marker_scales: dict[str, str] = field(default_factory=dict)


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def normalize_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the canonical upload contract."""
    required = {"subject", "condition", "epoch"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("missing required columns: " + ", ".join(sorted(missing)))
    out = frame.copy()
    if "region" not in out:
        out["region"] = "whole_scalp"
    out["subject"] = out["subject"].astype(str)
    out["condition"] = out["condition"].astype(str)
    out["region"] = out["region"].fillna("whole_scalp").astype(str)
    out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce")
    out = out.dropna(subset=["epoch"])
    if out["condition"].nunique() < 2:
        raise ValueError("at least two conditions are required")
    numeric_markers = [c for c in out.columns if c not in META_COLUMNS
                       and pd.api.types.is_numeric_dtype(out[c])]
    if not numeric_markers:
        raise ValueError("no numeric marker columns found")
    return out.reset_index(drop=True)


def music_bundle(frame: pd.DataFrame | None = None) -> DatasetBundle:
    epochs = _read_csv(MUSIC_PATH) if frame is None else frame.copy()
    if len(epochs):
        epochs = normalize_table(epochs)
    return DatasetBundle(
        "music", "ICU music", epochs, default_condition="Pink Noise",
        default_reference="No Music", default_marker="theta_rel")


def openneuro_bundle(frame: pd.DataFrame | None = None) -> DatasetBundle:
    epochs = _read_csv(OPENNEURO_PATH) if frame is None else frame.copy()
    if len(epochs):
        required = {"subject", "state", "epoch", "block"}
        missing = required - set(epochs.columns)
        if missing:
            raise ValueError("OpenNeuro table missing: " + ", ".join(sorted(missing)))
        # The long first eyes-closed block is a resting baseline, not part of the alternating
        # eyes protocol used for the paired Berger effect.
        first = epochs.groupby("subject")["block"].transform("min")
        epochs = epochs.loc[epochs["block"] != first].copy()
        epochs["condition"] = epochs["state"].map(
            {"eyes_closed": "Eyes closed", "eyes_open": "Eyes open"}).fillna(
                epochs["state"].astype(str))
        epochs = normalize_table(epochs)
    return DatasetBundle(
        "openneuro", "OpenNeuro eyes protocol", epochs, block_col="block",
        floor_factor=1.0, default_condition="Eyes closed", default_reference="Eyes open",
        default_marker="adr")


# Hosted demos must never read local research presets, even when paths are configured.
MUSIC = music_bundle(pd.DataFrame()) if os.environ.get("FEASIBILITY_PUBLIC_DEMO") == "1" else music_bundle()
OPENNEURO = openneuro_bundle(pd.DataFrame()) if os.environ.get("FEASIBILITY_PUBLIC_DEMO") == "1" else openneuro_bundle()


def parse_uploaded_csv(contents: str | None, filename: str | None = None) -> dict | None:
    if not contents:
        return None
    if filename and not filename.lower().endswith((".csv", ".txt")):
        raise ValueError("upload a CSV file")
    try:
        _, encoded = contents.split(",", 1)
        raw = base64.b64decode(encoded)
        frame = pd.read_csv(io.BytesIO(raw))
        frame = normalize_table(frame)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not read CSV: {exc}") from exc
    return {"filename": filename or "uploaded.csv", "table": table_json(frame)}


def table_json(frame: pd.DataFrame) -> str:
    """Preserve float precision, including EEG powers much smaller than 1e-10."""
    values = frame.astype(object).where(pd.notna(frame), None)
    return json.dumps(values.to_dict(orient="split"), allow_nan=False)


def parse_tabular_upload(contents: str | None, filename: str | None = None) -> dict | None:
    """Read an unstaged CSV/TSV/XLSX without assuming canonical column names."""
    if not contents:
        return None
    name = filename or "uploaded.csv"
    extension = os.path.splitext(name.lower())[1]
    if extension not in {".csv", ".tsv", ".txt", ".xlsx"}:
        raise ValueError("upload a CSV, TSV, TXT, or XLSX table")
    try:
        _, encoded = contents.split(",", 1)
        raw = base64.b64decode(encoded)
        if extension == ".xlsx":
            frame = pd.read_excel(io.BytesIO(raw), dtype=str)
        else:
            separator = "\t" if extension == ".tsv" else None
            frame = pd.read_csv(io.BytesIO(raw), sep=separator, engine="python", dtype=str)
    except Exception as exc:
        raise ValueError(f"could not read table: {exc}") from exc
    if frame.empty:
        raise ValueError("the uploaded table has no rows")
    if len(frame.columns) < 3:
        raise ValueError("the uploaded table needs at least three columns")
    return {"filename": name, "table": table_json(frame)}


def parse_recipe_upload(contents: str | None, filename: str | None = None) -> dict | None:
    """Decode a Dash JSON upload and validate its recipe schema."""
    if not contents:
        return None
    if filename and not filename.lower().endswith(".json"):
        raise ValueError("upload a recipe JSON file")
    try:
        _, encoded = contents.split(",", 1)
        raw = base64.b64decode(encoded).decode("utf-8")
        recipe = parse_recipe(raw)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not read recipe JSON: {exc}") from exc
    return {"filename": filename or "staging-recipe.json", "recipe": recipe}


def raw_uploaded_frame(payload: dict | None) -> pd.DataFrame:
    if not payload or "table" not in payload:
        return pd.DataFrame()
    return pd.read_json(io.StringIO(payload["table"]), orient="split", dtype=False,
                        precise_float=True)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip()).strip("_").lower()
    return slug or "derived_marker"


def _numeric_measurement(values: pd.Series, name: str) -> pd.Series:
    """Allow missing observations, but reject invalid values instead of silently dropping them."""
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = (values.notna() & numeric.isna()) | np.isinf(numeric)
    if invalid.any():
        raise ValueError(f"{name}: {int(invalid.sum())} nonnumeric or infinite values; correct the column")
    if not numeric.notna().any():
        raise ValueError(f"{name}: no usable numeric measurements")
    return numeric


def stage_table(
    frame: pd.DataFrame,
    *,
    subject_col: str,
    condition_col: str,
    epoch_col: str | None,
    region_col: str | None,
    block_col: str | None,
    marker_cols: list[str],
    log_cols: list[str] | None = None,
    relative_cols: list[str] | None = None,
    derived_name: str | None = None,
    derived_operation: str | None = None,
    derived_a: str | None = None,
    derived_inputs: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map an arbitrary repeated-measures table onto the canonical analysis contract."""
    required_roles = {subject_col, condition_col}
    if not subject_col or not condition_col or not required_roles <= set(frame.columns):
        raise ValueError("map both the subject and condition columns")
    if subject_col == condition_col:
        raise ValueError("subject and condition must use different columns")
    assigned_roles = [col for col in (subject_col, condition_col, epoch_col, region_col, block_col)
                      if col]
    if len(assigned_roles) != len(set(assigned_roles)):
        raise ValueError("each dataset role must use a different column")
    missing_roles = set(assigned_roles) - set(frame.columns)
    if missing_roles:
        raise ValueError("mapped columns not found: " + ", ".join(sorted(missing_roles)))
    for role in (subject_col, condition_col):
        if frame[role].isna().any() or frame[role].astype(str).str.strip().eq("").any():
            raise ValueError(f"{role}: subject and condition values must not be missing")
    marker_cols = list(dict.fromkeys(marker_cols or []))
    if not marker_cols:
        raise ValueError("select at least one numeric measurement column")
    missing = set(marker_cols) - set(frame.columns)
    if missing:
        raise ValueError("measurement columns not found: " + ", ".join(sorted(missing)))
    role_overlap = set(marker_cols) & set(assigned_roles)
    if role_overlap:
        raise ValueError("role columns cannot also be measurements: " +
                         ", ".join(sorted(role_overlap)))
    reserved = set(marker_cols) & META_COLUMNS
    if reserved:
        raise ValueError("rename measurements using reserved role names: " +
                         ", ".join(sorted(reserved)))

    chosen = [subject_col, condition_col]
    for optional in (epoch_col, region_col, block_col):
        if optional and optional not in chosen:
            chosen.append(optional)
    chosen.extend(col for col in marker_cols if col not in chosen)
    out = frame.loc[:, chosen].copy()
    rename = {subject_col: "subject", condition_col: "condition"}
    if epoch_col:
        rename[epoch_col] = "epoch"
    if region_col:
        rename[region_col] = "region"
    if block_col:
        rename[block_col] = "block"
    if len({rename.get(col, col) for col in chosen}) != len(chosen):
        raise ValueError("mapped column names collide; rename the conflicting source columns")
    out = out.rename(columns=rename)
    staged_markers = [rename.get(col, col) for col in marker_cols]
    if len(set(staged_markers)) != len(staged_markers):
        raise ValueError("mapped measurement names must be unique")
    for marker in staged_markers:
        out[marker] = _numeric_measurement(out[marker], marker)

    if "epoch" not in out:
        group = ["subject", "condition"] + (["region"] if "region" in out else [])
        out["epoch"] = out.groupby(group, dropna=False).cumcount()
    if "region" not in out:
        out["region"] = "whole_scalp"
    for role in ("epoch", "block"):
        if role in out:
            numeric = pd.to_numeric(out[role], errors="coerce")
            if not np.isfinite(numeric).all():
                raise ValueError(f"{role}: map a complete numeric order column")
            out[role] = numeric
    keys = ["subject", "condition", "region"] + (["block"] if "block" in out else []) + ["epoch"]
    if out.duplicated(keys).any():
        raise ValueError("duplicate measurement keys; map the block/region or a unique epoch order")

    log_set = {rename.get(col, col) for col in (log_cols or [])}
    relative_set = {rename.get(col, col) for col in (relative_cols or [])}
    if (log_set | relative_set) - set(staged_markers):
        raise ValueError("scale settings must refer to selected measurements")
    overlap = log_set & relative_set
    if overlap:
        raise ValueError("a measurement cannot be both log-scaled and relative power")
    scales = {marker: ("log10" if marker in log_set else
                       "relative" if marker in relative_set else "raw")
              for marker in staged_markers}
    for marker in log_set:
        if out[marker].dropna().le(0).any():
            raise ValueError(f"{marker}: log10 requires positive measurements")

    if derived_operation:
        if not derived_a or derived_a not in frame:
            raise ValueError("choose the primary input for the derived marker")
        inputs = list(dict.fromkeys(derived_inputs or []))
        if ({derived_a} | set(inputs)) & set(assigned_roles):
            raise ValueError("derived inputs must be measurements, not mapped role columns")
        a = _numeric_measurement(frame[derived_a], derived_a)
        name = _slug(derived_name or derived_operation)
        if name in META_COLUMNS or name in out.columns:
            raise ValueError(f"derived marker name '{name}' is already in use")
        if derived_operation == "log10":
            if a.dropna().le(0).any():
                raise ValueError("log10 requires positive input values")
            out[name] = np.log10(a.where(a > 0))
            scales[name] = "log10_value"  # values are already transformed
        else:
            if not inputs or any(col not in frame for col in inputs):
                raise ValueError("choose the comparison input columns for the derived marker")
            if derived_operation in ("ratio", "difference") and len(inputs) != 1:
                raise ValueError("ratio and difference require exactly one comparison input")
            numeric = [_numeric_measurement(frame[col], col) for col in inputs]
            if derived_operation == "ratio":
                denominator = numeric[0]
                out[name] = a / denominator.where(denominator != 0)
                scales[name] = "log10"
            elif derived_operation == "difference":
                out[name] = a - numeric[0]
                scales[name] = "raw"
            elif derived_operation == "share":
                denominator = sum(numeric)
                out[name] = a / denominator.where(denominator != 0)
                scales[name] = "relative"
            else:
                raise ValueError("unknown derived-marker operation")
            if derived_operation in ("ratio", "share") and denominator.dropna().eq(0).any():
                raise ValueError("derived marker denominator contains zero; correct the inputs")
        out[name] = _numeric_measurement(out[name], name)
        if scales[name] == "log10" and out[name].dropna().le(0).any():
            raise ValueError("derived ratio requires positive values for log10 comparison")

    out = normalize_table(out)
    return out, scales


def staged_payload(frame: pd.DataFrame, filename: str, **kwargs) -> dict:
    epoch_seconds = float(kwargs.pop("epoch_seconds", EPOCH_SECONDS))
    if not np.isfinite(epoch_seconds) or epoch_seconds <= 0:
        raise ValueError("epoch duration must be greater than zero")
    staged, scales = stage_table(frame, **kwargs)
    return {
        "ok": True, "filename": filename, "table": table_json(staged),
        "marker_scales": scales, "epoch_seconds": epoch_seconds,
    }


def uploaded_bundle(payload: dict | None) -> DatasetBundle:
    if not payload or "table" not in payload:
        return DatasetBundle("upload", "Uploaded CSV", pd.DataFrame())
    frame = pd.read_json(io.StringIO(payload["table"]), orient="split", dtype=False,
                         precise_float=True)
    frame = normalize_table(frame)
    block_col = "block" if "block" in frame.columns else None
    return DatasetBundle(
        "upload", payload.get("filename", "Uploaded table"), frame,
        epoch_seconds=float(payload.get("epoch_seconds", EPOCH_SECONDS)), block_col=block_col,
        floor_factor=1.0 if block_col else 1 / np.sqrt(2.0),
        marker_scales=payload.get("marker_scales", {}))


def get_bundle(source: str, payload: dict | None = None) -> DatasetBundle:
    if source == "openneuro":
        return OPENNEURO
    if source == "upload":
        return uploaded_bundle(payload)
    return MUSIC


def label(marker: str) -> str:
    return MARKER_LABELS.get(marker, marker.replace("_", " "))


def uses_log(marker: str, bundle: DatasetBundle | None = None) -> bool:
    if bundle and marker in bundle.marker_scales:
        return bundle.marker_scales[marker] == "log10"
    return rx.uses_log(marker) or marker.endswith("_abs")


def units(marker: str, bundle: DatasetBundle | None = None) -> str:
    if uses_log(marker, bundle):
        return "log10 units"
    if bundle and bundle.marker_scales.get(marker) == "log10_value":
        return "log10 units"
    if bundle and bundle.marker_scales.get(marker) == "relative":
        return "relative-power proportion"
    if bundle and marker in bundle.marker_scales:
        return "marker units"
    if marker.endswith("_rel"):
        return "relative-power proportion"
    return "marker units"


def markers_available(frame: pd.DataFrame) -> list[str]:
    found = [c for c in frame.columns if c not in META_COLUMNS
             and pd.api.types.is_numeric_dtype(frame[c])]
    ordered = [m for m in MARKER_ORDER if m in found]
    return ordered + sorted(m for m in found if m not in ordered)


def regions_available(frame: pd.DataFrame) -> list[str]:
    if not len(frame) or "region" not in frame:
        return ["whole_scalp"]
    found = set(frame["region"].dropna().astype(str))
    ordered = [r for r in REGION_ORDER if r in found]
    return ordered + sorted(found - set(ordered)) or ["whole_scalp"]


def conditions_available(frame: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(frame.get("condition", pd.Series(dtype=str)).dropna().astype(str)))


def selected(bundle: DatasetBundle, marker: str, region: str) -> pd.DataFrame:
    frame = bundle.epochs
    if not len(frame) or marker not in frame or region not in set(frame["region"]):
        return pd.DataFrame(columns=["subject", "condition", "epoch", "value"])
    cols = ["subject", "condition", "epoch", "region"]
    if bundle.block_col and bundle.block_col in frame:
        cols.append(bundle.block_col)
    out = frame.loc[frame["region"] == region, cols].copy()
    values = pd.to_numeric(frame.loc[frame["region"] == region, marker], errors="coerce")
    if uses_log(marker, bundle):
        values = np.log10(values.where(values > 0))
    out["value"] = values.to_numpy()
    return out.dropna(subset=["value"]).reset_index(drop=True)


def contrast_deltas(bundle: DatasetBundle, marker: str, region: str,
                    condition: str, reference: str) -> pd.Series:
    if condition == reference:
        return pd.Series(dtype=float, name="delta")
    sel = selected(bundle, marker, region)
    if bundle.block_col and bundle.block_col in sel:
        # A block is the repeat unit in alternating protocols.  Give blocks equal weight rather
        # than letting a block with one extra usable epoch pull the subject's condition mean.
        block = bundle.block_col
        sel = (sel.groupby(["subject", "condition", block], as_index=False)["value"].mean())
    means = sel.groupby(["subject", "condition"])["value"].mean().unstack("condition")
    if condition not in means or reference not in means:
        return pd.Series(dtype=float, name="delta")
    return (means[condition] - means[reference]).dropna().rename("delta")


def group_stats(delta: pd.Series) -> dict:
    """Group statistics with an explicit zero-variance guard for arbitrary uploads."""
    values = pd.to_numeric(delta, errors="coerce").dropna().to_numpy()
    if len(values) < 2:
        return {"n": len(values)}
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    tolerance = max(1.0, abs(mean) if np.isfinite(mean) else 1.0) * 1e-12
    if np.isfinite(sd) and sd <= tolerance:
        # Constant paired changes have no finite standardized effect.  Do not display a
        # floating-point approximation to infinity as if it were an ordinary dz.
        return {"n": len(values), "mean": mean, "sd": sd, "cohen_dz": float("nan")}
    return rx.group_stats(pd.Series(values))


def effect_table(bundle: DatasetBundle, region: str, condition: str,
                 reference: str) -> pd.DataFrame:
    rows = []
    for marker in markers_available(bundle.epochs):
        stats = group_stats(contrast_deltas(bundle, marker, region, condition, reference))
        if stats.get("n", 0) >= 2 and np.isfinite(stats.get("cohen_dz", np.nan)):
            rows.append({"marker": marker, **stats})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_wilcoxon"] = rx.bh_fdr(out["p_wilcoxon"])
    return out


def floor_differences(bundle: DatasetBundle, marker: str, region: str,
                      condition: str, reference: str) -> pd.DataFrame:
    sel = selected(bundle, marker, region)
    sel = sel[sel["condition"].isin([condition, reference])]
    if not len(sel):
        return pd.DataFrame(columns=["subject", "state", "floor_diff", "n_epochs"])
    if bundle.block_col and bundle.block_col in sel:
        block = bundle.block_col
        bm = (sel.groupby(["subject", "condition", block])["value"]
              .agg([("value", "mean"), ("n_epochs", "size")]).reset_index())
        rows = []
        for (subject, state), group in bm.groupby(["subject", "condition"]):
            group = group.sort_values(block)
            values = group["value"].to_numpy()
            sizes = group["n_epochs"].to_numpy()
            for i, (a, b) in enumerate(zip(values[:-1], values[1:])):
                rows.append({"subject": subject, "state": state, "floor_diff": b - a,
                             "n_epochs": int(min(sizes[i], sizes[i + 1]))})
        return pd.DataFrame(rows)
    frames = []
    for state in dict.fromkeys([reference, condition]):
        floors = rx.noise_floor(sel, reference=state, scheme="halves")
        if len(floors):
            floors.insert(1, "state", state)
            frames.append(floors)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["subject", "state", "floor_diff", "n_epochs"])


def contrast_halves(bundle: DatasetBundle, marker: str, region: str,
                    condition: str, reference: str) -> pd.DataFrame:
    sel = selected(bundle, marker, region)
    sel = sel[sel["condition"].isin([condition, reference])].copy()
    if not len(sel) or condition == reference:
        return pd.DataFrame(columns=["half_odd", "half_even"])
    if bundle.block_col and bundle.block_col in sel:
        block = bundle.block_col
        values = (sel.groupby(["subject", "condition", block])["value"].mean().reset_index()
                  .sort_values(["subject", "condition", block]))
        values["parity"] = values.groupby(["subject", "condition"]).cumcount() % 2
    else:
        values = sel.sort_values(["subject", "condition", "epoch"])
        values["parity"] = values.groupby(["subject", "condition"]).cumcount() % 2
    halves = {}
    for parity, name in ((0, "half_odd"), (1, "half_even")):
        means = values[values["parity"] == parity].groupby(
            ["subject", "condition"])["value"].mean().unstack("condition")
        if condition not in means or reference not in means:
            return pd.DataFrame(columns=["half_odd", "half_even"])
        halves[name] = (means[condition] - means[reference]).dropna()
    idx = halves["half_odd"].index.intersection(halves["half_even"].index)
    return pd.DataFrame({name: series.loc[idx].to_numpy() for name, series in halves.items()},
                        index=idx)


def reference_epoch_count(bundle: DatasetBundle, marker: str, region: str,
                          condition: str, reference: str) -> float:
    sel = selected(bundle, marker, region)
    sel = sel[sel["condition"].isin([condition, reference])]
    if not len(sel):
        return float("nan")
    group = ["subject", "condition"]
    if bundle.block_col and bundle.block_col in sel:
        group.append(bundle.block_col)
    return float(sel.groupby(group).size().median())


def required_n(dz: float) -> float:
    if not np.isfinite(dz) or dz == 0:
        return float("nan")
    return float(np.ceil((Z_975 + Z_POWER_80) ** 2 / dz ** 2))


def measurement(bundle: DatasetBundle, marker: str, region: str, condition: str,
                reference: str, target_fraction: float = 1.0) -> dict:
    deltas = contrast_deltas(bundle, marker, region, condition, reference)
    effect = group_stats(deltas)
    floors = floor_differences(bundle, marker, region, condition, reference)
    floor_stats = rx.floor_summary(floors) if len(floors) else {"n": 0}
    halves = contrast_halves(bundle, marker, region, condition, reference)
    reliability = (rx.split_half_reliability(halves["half_odd"], halves["half_even"])
                   if len(halves) else {"n": 0})
    n_ref = reference_epoch_count(bundle, marker, region, condition, reference)
    floor = floor_stats.get("abs_p95", float("nan")) * bundle.floor_factor
    mean, sd = effect.get("mean", float("nan")), effect.get("sd", float("nan"))
    target = abs(mean) * float(target_fraction) if np.isfinite(mean) else float("nan")
    epochs_at_target = (n_ref * (floor / target) ** 2 if np.isfinite(target) and target > 0
                        and np.isfinite(floor) else float("nan"))
    dz_target = target / sd if np.isfinite(sd) and sd > 0 else float("nan")
    clears = int((deltas.abs() > floor).sum()) if len(deltas) and np.isfinite(floor) else 0
    return {
        "deltas": deltas, "effect": effect, "floors": floors, "floor": floor_stats,
        "halves": halves, "reliability": reliability, "n_ref": n_ref,
        "floor_matched": floor, "target": target, "epochs_at_target": epochs_at_target,
        "minutes_at_target": epochs_at_target * bundle.epoch_seconds / 60.0,
        "required_n": required_n(dz_target), "n_clears": clears,
    }


def value_text(marker: str, value: float, signed: bool = True,
               bundle: DatasetBundle | None = None) -> str:
    if not np.isfinite(value):
        return "not estimable"
    sign = "+" if signed else ""
    if uses_log(marker, bundle):
        return f"{value:{sign}.3f} log10 ({10 ** value:.2f}x)"
    if bundle and bundle.marker_scales.get(marker) == "log10_value":
        return f"{value:{sign}.3f} log10"
    if bundle and bundle.marker_scales.get(marker) == "relative":
        return f"{value:{sign}.3f} ({value * 100:{sign}.1f} percentage points)"
    if bundle and marker in bundle.marker_scales:
        return f"{value:{sign}.3f}"
    if marker.endswith("_rel"):
        return f"{value:{sign}.3f} ({value * 100:{sign}.1f} percentage points)"
    return f"{value:{sign}.3f}"


def _blank(message: str, height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=.5, y=.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=13, color=MUTED))
    fig.update_layout(height=height, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      paper_bgcolor=WHITE, plot_bgcolor=WHITE,
                      margin=dict(l=30, r=30, t=55, b=35))
    return fig


def _chart_layout(fig: go.Figure, title: str, x_title: str,
                  y_title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(family="Iowan Old Style, Charter, Georgia, serif", size=19)),
        xaxis_title=x_title, yaxis_title=y_title, height=height, paper_bgcolor=WHITE,
        plot_bgcolor=WHITE, font=dict(family="Avenir Next, Segoe UI, sans-serif", color=INK),
        margin=dict(l=72, r=34, t=62, b=62), showlegend=False, title_x=.13)
    fig.update_xaxes(gridcolor=MIST, zerolinecolor=TRACE_LIGHT)
    fig.update_yaxes(gridcolor=MIST, zerolinecolor=TRACE_LIGHT)
    return fig


def fig_condition_overview(bundle: DatasetBundle, marker: str, region: str,
                           reference: str) -> go.Figure:
    rows = []
    for condition in conditions_available(bundle.epochs):
        if condition == reference:
            continue
        stats = group_stats(contrast_deltas(bundle, marker, region, condition, reference))
        if (stats.get("n", 0) >= 2 and stats.get("sd", 0) > 0
                and np.isfinite(stats.get("cohen_dz", np.nan))):
            rows.append({"condition": condition, **stats})
    if not rows:
        return _blank("No paired conditions available for this reference.", 300)
    table = pd.DataFrame(rows)
    table["q"] = rx.bh_fdr(table["p_wilcoxon"])
    table["lo_d"] = table["ci_lo"] / table["sd"]
    table["hi_d"] = table["ci_hi"] / table["sd"]
    table = table.sort_values("cohen_dz", key=lambda s: s.abs())
    fig = go.Figure()
    for _, row in table.iterrows():
        fig.add_scatter(
            x=[row["cohen_dz"]], y=[row["condition"]], mode="markers",
            marker=dict(size=11, color=TRACE),
            error_x=dict(type="data", symmetric=False, array=[row["hi_d"] - row["cohen_dz"]],
                         arrayminus=[row["cohen_dz"] - row["lo_d"]], color=TRACE_LIGHT),
            hovertemplate=(f"{row['condition']} minus {reference}<br>dz {row['cohen_dz']:+.2f}"
                           f"<br>q {row['q']:.3f}<br>n {int(row['n'])}<extra></extra>"))
        fig.add_annotation(x=row["cohen_dz"], y=row["condition"], showarrow=False,
                           xanchor="left" if row["cohen_dz"] >= 0 else "right",
                           text=f"  dz {row['cohen_dz']:+.2f} · q {row['q']:.3f}",
                           font=dict(size=10, color=MUTED))
    fig.add_vline(x=0, line_color=TRACE_LIGHT, line_width=1)
    span = max(.6, float(np.nanmax(np.abs(table[["lo_d", "hi_d"]].to_numpy()))) * 1.55)
    _chart_layout(fig, f"Conditions versus {reference} · {label(marker)}",
                  "Cohen's dz (unitless)", height=300)
    fig.update_xaxes(range=[-span, span])
    fig.update_layout(margin=dict(l=150, r=42, t=62, b=70))
    return fig


def fig_effects(bundle: DatasetBundle, marker: str, region: str,
                condition: str, reference: str, table: pd.DataFrame | None = None) -> go.Figure:
    table = effect_table(bundle, region, condition, reference) if table is None else table
    if not len(table):
        return _blank("No paired subjects for this comparison.")
    table = table.copy()
    table["lo_d"] = table["ci_lo"] / table["sd"]
    table["hi_d"] = table["ci_hi"] / table["sd"]
    table = table.sort_values("cohen_dz", key=lambda s: s.abs())
    fig = go.Figure()
    for _, row in table.iterrows():
        chosen = row["marker"] == marker
        colour = TRACE if chosen else TRACE_LIGHT
        fig.add_scatter(
            x=[row["cohen_dz"]], y=[label(row["marker"])], mode="markers",
            marker=dict(size=13 if chosen else 9, color=colour,
                        line=dict(width=2 if chosen else 0, color=INK)),
            error_x=dict(type="data", symmetric=False, array=[row["hi_d"] - row["cohen_dz"]],
                         arrayminus=[row["cohen_dz"] - row["lo_d"]], color=colour),
            hovertemplate=(f"{label(row['marker'])}<br>dz {row['cohen_dz']:+.2f}"
                           f"<br>q {row['q_wilcoxon']:.3f}<br>n {int(row['n'])}<extra></extra>"))
    fig.add_vline(x=0, line_color=TRACE_LIGHT, line_width=1)
    span = max(.6, float(np.nanmax(np.abs(table[["lo_d", "hi_d"]].to_numpy()))) * 1.5)
    _chart_layout(fig, f"Marker screen · {condition} minus {reference}",
                  "Cohen's dz (unitless)", height=410)
    fig.update_xaxes(range=[-span, span])
    fig.update_layout(margin=dict(l=145, r=38, t=62, b=70))
    return fig


def fig_floor(bundle: DatasetBundle, marker: str, region: str, condition: str,
              reference: str, measured: dict | None = None) -> go.Figure:
    measured = measurement(bundle, marker, region, condition, reference) if measured is None else measured
    floors = measured["floors"]
    if not len(floors):
        return _blank("No repeated same-condition measurements available.")
    mean, threshold = measured["effect"].get("mean", np.nan), measured["floor_matched"]
    fig = go.Figure()
    fig.add_histogram(x=floors["floor_diff"], nbinsx=42, marker_color="#c7d7dd",
                      marker_line=dict(color=TRACE_LIGHT, width=.5),
                      hovertemplate="same-condition change %{x:.3f}<br>count %{y}<extra></extra>")
    for sign in (-1, 1):
        fig.add_vline(x=sign * threshold, line_color=AMBER, line_dash="dot", line_width=2)
    fig.add_vline(x=mean, line_color=TRACE, line_width=3)
    _chart_layout(fig, f"Same-condition floor · {label(marker)}",
                  f"same-condition change ({units(marker, bundle)})", "repeated pairs", 380)
    return fig


def fig_reliability(bundle: DatasetBundle, marker: str, region: str, condition: str,
                    reference: str, measured: dict | None = None) -> go.Figure:
    measured = measurement(bundle, marker, region, condition, reference) if measured is None else measured
    halves, rel = measured["halves"], measured["reliability"]
    if len(halves) < 3:
        return _blank("Too few paired subjects for split-half reliability.")
    values = halves[["half_odd", "half_even"]].to_numpy()
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = (hi - lo) * .08 or .1
    fig = go.Figure()
    fig.add_scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
                    line=dict(color=TRACE_LIGHT, dash="dash", width=1), hoverinfo="skip")
    fig.add_scatter(x=halves["half_odd"], y=halves["half_even"], mode="markers",
                    marker=dict(color=TRACE, size=8, opacity=.66,
                                line=dict(color=WHITE, width=.5)),
                    hovertemplate="first index %{x:.3f}<br>second index %{y:.3f}<extra></extra>")
    fig.add_annotation(x=.03, y=.97, xref="paper", yref="paper", xanchor="left", yanchor="top",
                       showarrow=False, align="left", font=dict(size=11, color=INK),
                       text=(f"Pearson r {rel.get('pearson_r', np.nan):+.2f}<br>"
                             f"Spearman–Brown {rel.get('spearman_brown', np.nan):.2f}<br>"
                             f"n = {rel.get('n', 0)}"))
    _chart_layout(fig, f"Contrast reliability · {label(marker)}",
                  f"first internal index ({units(marker, bundle)})",
                  f"second internal index ({units(marker, bundle)})", 390)
    fig.update_xaxes(range=[lo - pad, hi + pad])
    fig.update_yaxes(range=[lo - pad, hi + pad])
    return fig


def fig_detectability(bundle: DatasetBundle, marker: str, region: str, condition: str,
                      reference: str, target_fraction: float = 1.0,
                      measured: dict | None = None) -> go.Figure:
    measured = (measurement(bundle, marker, region, condition, reference, target_fraction)
                if measured is None else measured)
    floor, n_ref, target = (measured["floor_matched"], measured["n_ref"], measured["target"])
    if not all(np.isfinite(x) and x > 0 for x in (floor, n_ref, target)):
        return _blank("Detectability cannot be estimated for this selection.")
    n_max = min(4000., max(240., 2 * measured["epochs_at_target"]))
    grid = np.geomspace(max(2., n_ref / 4), n_max, 220)
    minutes = grid * bundle.epoch_seconds / 60
    curve = floor * np.sqrt(n_ref / grid)
    fig = go.Figure()
    fig.add_scatter(x=minutes, y=curve, mode="lines", line=dict(color=TRACE, width=3),
                    hovertemplate="%{x:.1f} min/unit<br>detectable change %{y:.3f}<extra></extra>")
    fig.add_hline(y=target, line_color=AMBER, line_dash="dash", line_width=2)
    fig.add_vline(x=n_ref * bundle.epoch_seconds / 60, line_color=TRACE_LIGHT, line_width=1)
    cross = measured["minutes_at_target"]
    if np.isfinite(cross) and minutes.min() <= cross <= minutes.max():
        fig.add_scatter(x=[cross], y=[target], mode="markers",
                        marker=dict(size=12, symbol="circle-open", color=AMBER,
                                    line=dict(width=2.5)), hoverinfo="skip")
    _chart_layout(fig, f"Recording length · {label(marker)}", "minutes per measurement unit",
                  f"smallest detectable change ({units(marker, bundle)})", 390)
    fig.update_layout(margin=dict(l=88, r=34, t=62, b=62), title_x=.16)
    fig.update_xaxes(type="log", range=[np.log10(minutes.min() * .9),
                                         np.log10(minutes.max() * 1.1)])
    y_lo, y_hi = min(curve.min(), target), max(curve.max(), target)
    fig.update_yaxes(type="log", range=[np.log10(y_lo * .8), np.log10(y_hi * 1.25)])
    return fig


def audit_readout(bundle: DatasetBundle, marker: str, region: str, condition: str,
                  reference: str, target_fraction: float, measured: dict | None = None) -> list[tuple[str, str]]:
    measured = (measurement(bundle, marker, region, condition, reference, target_fraction)
                if measured is None else measured)
    effect, rel = measured["effect"], measured["reliability"]
    n = effect.get("n", 0)
    return [
        ("Paired cohort", f"{n} subjects"),
        ("Observed mean", value_text(marker, effect.get("mean", np.nan), bundle=bundle)),
        ("Cohen's dz", (f"{effect.get('cohen_dz', np.nan):+.2f}"
                         if n >= 2 and np.isfinite(effect.get("cohen_dz", np.nan))
                         else "not estimable")),
        ("p95 floor", value_text(marker, measured["floor_matched"], signed=False,
                                  bundle=bundle)),
        ("Clear floor individually", f"{measured['n_clears']}/{n}"
         if n and np.isfinite(measured["floor_matched"]) else "not estimable"),
        ("Spearman–Brown", (f"{rel.get('spearman_brown', np.nan):.2f}"
                             if rel.get("n", 0) and np.isfinite(rel.get("spearman_brown", np.nan))
                             else "not estimable")),
        ("Minutes for target", f"{measured['minutes_at_target']:.1f}" if np.isfinite(measured["minutes_at_target"]) else "not estimable"),
        ("Approximate n for 80% power", f"{measured['required_n']:.0f}" if np.isfinite(measured["required_n"]) else "not estimable"),
    ]


def _options(values: list[str], transform=lambda x: x) -> list[dict]:
    return [{"label": transform(value), "value": value} for value in values]


NONE_COLUMN = "__not_used__"


def _guess_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lowered = {str(col).strip().lower().replace(" ", "_"): col for col in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    for alias in aliases:
        match = next((col for key, col in lowered.items() if alias in key), None)
        if match is not None:
            return match
    return None


def staging_defaults(frame: pd.DataFrame) -> dict:
    """Infer conservative initial role assignments while leaving the user in control."""
    columns = list(frame.columns)
    roles = {
        "subject": _guess_column(columns, ("subject", "participant", "patient", "subject_id",
                                            "participant_id", "patient_id", "id")),
        "condition": _guess_column(columns, ("condition", "state", "treatment", "arm",
                                              "stimulus", "group")),
        "epoch": _guess_column(columns, ("epoch", "trial", "window", "segment", "timepoint")),
        "region": _guess_column(columns, ("region", "roi", "sensor_region", "channel_group")),
        "block": _guess_column(columns, ("block", "run", "session", "visit")),
    }
    role_columns = {value for value in roles.values() if value is not None}
    numeric = []
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if column not in role_columns and converted.notna().mean() >= .8:
            numeric.append(column)
    return {
        **roles,
        "markers": numeric,
        "log": [col for col in numeric if uses_log(str(col))],
        "relative": [col for col in numeric if str(col).lower().endswith("_rel")
                     or "relative" in str(col).lower()],
    }


def staging_numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric-like columns available for measurements and formula operands."""
    return [column for column in frame.columns
            if pd.to_numeric(frame[column], errors="coerce").notna().any()]


def staging_summary(frame: pd.DataFrame) -> str:
    counts = frame.groupby(["subject", "condition"], dropna=False).size()
    repeated = bool(len(counts) and counts.max() >= 2)
    floor = "repeated measurements available" if repeated else "paired contrast only"
    return (f"Ready · {len(frame):,} rows · {frame['subject'].nunique()} subjects · "
            f"{frame['condition'].nunique()} conditions · "
            f"{len(markers_available(frame))} measurements · {floor}")


def _preview_table(frame: pd.DataFrame):
    if frame.empty:
        return "Choose a table to inspect its columns."
    preview = frame.head(6).fillna("")
    return html.Div(className="preview-wrap", children=html.Table([
        html.Thead(html.Tr([html.Th(str(col)) for col in preview.columns])),
        html.Tbody([html.Tr([html.Td(str(value)) for value in row])
                    for row in preview.itertuples(index=False, name=None)]),
    ]))


def configuration(bundle: DatasetBundle) -> tuple:
    if not len(bundle.epochs):
        return [], None, [], None, [], None, [], None
    conditions = conditions_available(bundle.epochs)
    markers = markers_available(bundle.epochs)
    regions = regions_available(bundle.epochs)
    reference = bundle.default_reference if bundle.default_reference in conditions else conditions[0]
    condition = bundle.default_condition if bundle.default_condition in conditions else next(
        (x for x in conditions if x != reference), conditions[0])
    marker = bundle.default_marker if bundle.default_marker in markers else markers[0]
    region = "whole_scalp" if "whole_scalp" in regions else regions[0]
    return (_options(conditions), condition, _options(conditions), reference,
            _options(markers, label), marker,
            _options(regions, lambda x: x.replace("_", " ")), region)


app = Dash(__name__, title="Measurement feasibility")
app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
:root{--paper:#f6f8f9;--white:#fff;--ink:#17212b;--muted:#60717c;--trace:#1d5d78;
--mist:#e7eef1;--amber:#c6821e}*{box-sizing:border-box}body{margin:0;background:var(--paper);
color:var(--ink);font-family:"Avenir Next","Segoe UI",sans-serif}.shell{max-width:1280px;margin:0 auto;
padding:24px 30px 48px}.source,.controls{display:flex;flex-wrap:wrap;gap:14px;background:var(--white);
border:1px solid var(--mist);padding:18px;margin-bottom:16px}.source{align-items:end}.control{min-width:185px;flex:1}
.control.wide{min-width:320px;flex:2}.control label{display:block;font-size:11px;font-weight:700;
letter-spacing:.055em;text-transform:uppercase;margin:0 0 7px;color:var(--muted)}.upload{height:38px;
display:flex;align-items:center;justify-content:center;border:1px dashed #9babb4;color:var(--muted);
font-size:13px;cursor:pointer;padding:0 16px}.status{font:500 11px/1.45 SFMono-Regular,Consolas,monospace;
color:var(--muted);padding:2px 0}.audit{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));
background:var(--ink);color:white;margin-bottom:16px}.audit-cell{padding:14px 16px;border-right:1px solid #33414d;
border-bottom:1px solid #33414d}.audit-key{font:500 10px/1.2 SFMono-Regular,Consolas,monospace;
letter-spacing:.06em;text-transform:uppercase;color:#aebdc6}.audit-value{font-size:16px;margin-top:5px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}.panel{background:var(--white);
border:1px solid var(--mist);min-width:0}.overview{margin-bottom:16px}.Select-control,.upload{border-radius:0!important}
.stage{background:var(--white);border:1px solid var(--mist);margin:-1px 0 16px;padding:0 18px 18px}
.stage-head{display:flex;justify-content:space-between;gap:18px;align-items:baseline;padding:15px 0 12px;
border-bottom:1px solid var(--mist);margin-bottom:14px}.stage-file{font:600 12px/1.4 SFMono-Regular,Consolas,monospace}
.stage-grid{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:12px}.stage-grid+.stage-grid{margin-top:12px}
.stage .control{min-width:0}.formula{border-left:3px solid var(--trace);padding-left:13px}.preview-wrap{margin-top:14px;
max-height:210px;overflow:auto;border:1px solid var(--mist)}table{border-collapse:collapse;width:100%;font:500 11px/1.35 SFMono-Regular,Consolas,monospace}
th,td{padding:7px 10px;border-right:1px solid var(--mist);border-bottom:1px solid var(--mist);text-align:left;
white-space:nowrap}th{position:sticky;top:0;background:#edf2f4;color:var(--muted);z-index:1}.stage-actions{display:flex;
align-items:center;gap:14px;margin-top:14px}.stage-button{height:38px;border:0;background:var(--ink);color:white;padding:0 20px;
font-weight:650;cursor:pointer}.stage-button:hover{background:var(--trace)}.stage-result{font:500 11px/1.45 SFMono-Regular,Consolas,monospace;color:var(--muted)}
.stage-button:disabled{background:#9babb4;cursor:not-allowed}.recipe-upload{height:38px;display:flex;align-items:center;
justify-content:center;border:1px dashed #9babb4;color:var(--muted);font-size:12px;cursor:pointer;padding:0 16px}
*:focus-visible{outline:2px solid var(--trace);outline-offset:2px}@media(max-width:850px){.grid{grid-template-columns:1fr}
.audit{grid-template-columns:1fr 1fr}.stage-grid{grid-template-columns:1fr 1fr}.shell{padding:14px}}@media(max-width:520px){
.stage-grid{grid-template-columns:1fr}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""


def _control(text: str, component, wide: bool = False):
    return html.Div(className="control wide" if wide else "control",
                    children=[html.Label(text), component])


app.layout = html.Main(className="shell", children=[
    dcc.Store(id="raw-upload"),
    dcc.Store(id="uploaded-data"),
    dcc.Store(id="loaded-recipe"),
    dcc.Download(id="recipe-download"),
    html.Section(className="source", children=[
        _control("Dataset", dcc.Dropdown(
            id="dataset", options=_options(["music", "openneuro", "upload"],
                lambda x: {"music": "ICU music", "openneuro": "OpenNeuro eyes protocol",
                           "upload": "Upload & stage"}[x]),
            value="music", clearable=False)),
        _control("Table", dcc.Upload(id="upload", className="upload",
                 children="Choose table", accept=".csv,.tsv,.txt,.xlsx")),
        html.Div(id="source-status", className="status control wide"),
    ]),
    html.Section(id="stage", className="stage", children=[
        html.Div(className="stage-head", children=[
            html.Div("Column staging", className="stage-file"),
            html.Div(id="stage-file", className="status"),
        ]),
        html.Div(className="stage-grid", children=[
            _control("Subject", dcc.Dropdown(id="map-subject", clearable=False)),
            _control("Condition", dcc.Dropdown(id="map-condition", clearable=False)),
            _control("Epoch / order", dcc.Dropdown(id="map-epoch", clearable=False)),
            _control("Region", dcc.Dropdown(id="map-region", clearable=False)),
            _control("Block / numeric order", dcc.Dropdown(id="map-block", clearable=False)),
        ]),
        html.Div(className="stage-grid", children=[
            _control("Measurements", dcc.Dropdown(id="map-markers", multi=True), wide=True),
            _control("Compare on log10 scale", dcc.Dropdown(id="map-log", multi=True)),
            _control("Relative-power columns", dcc.Dropdown(id="map-relative", multi=True)),
            _control("Seconds per epoch", dcc.Input(id="epoch-seconds", type="number", min=.001,
                                                      step=.1, value=EPOCH_SECONDS)),
        ]),
        html.Div(className="stage-grid formula", children=[
            _control("Derived marker name", dcc.Input(id="derived-name", type="text",
                                                       placeholder="optional")),
            _control("Calculation", dcc.Dropdown(id="derived-operation", clearable=True,
                options=[{"label": "Ratio A / B", "value": "ratio"},
                         {"label": "Difference A − B", "value": "difference"},
                         {"label": "Share A / sum(inputs)", "value": "share"},
                         {"label": "Log10(A)", "value": "log10"}])),
            _control("Input A", dcc.Dropdown(id="derived-a", clearable=True)),
            _control("Input B / denominator set", dcc.Dropdown(id="derived-inputs", multi=True)),
        ]),
        html.Div(id="stage-preview"),
        html.Div(className="stage-actions", children=[
            html.Button("Stage dataset", id="stage-button", className="stage-button", n_clicks=0),
            html.Button("Download recipe", id="download-recipe-button", className="stage-button",
                        n_clicks=0, disabled=True),
            dcc.Upload(id="recipe-upload", className="recipe-upload", children="Load recipe JSON",
                       accept=".json"),
            html.Div(id="stage-result", className="stage-result"),
            html.Div(id="recipe-result", className="stage-result"),
        ]),
    ]),
    html.Div(id="analysis-workspace", children=[
        html.Section(className="controls", children=[
            _control("Condition", dcc.Dropdown(id="condition", clearable=False)),
            _control("Reference", dcc.Dropdown(id="reference", clearable=False)),
            _control("Marker", dcc.Dropdown(id="marker", clearable=False)),
            _control("Sensor region", dcc.Dropdown(id="region", clearable=False)),
            _control("Target fraction of observed effect", dcc.Slider(
                id="target", min=.25, max=2, step=.05, value=1,
                marks={.25: "¼", .5: "½", 1: "1×", 1.5: "1½", 2: "2×"}), wide=True),
        ]),
        html.Section(id="audit", className="audit"),
        html.Div(className="panel overview", children=dcc.Graph(id="g-overview")),
        html.Div(className="grid", children=[
            html.Div(className="panel", children=dcc.Graph(id="g-effects")),
            html.Div(className="panel", children=dcc.Graph(id="g-floor")),
            html.Div(className="panel", children=dcc.Graph(id="g-reliability")),
            html.Div(className="panel", children=dcc.Graph(id="g-detectability")),
        ]),
    ]),
])


@app.callback(Output("raw-upload", "data"),
              Output("uploaded-data", "data", allow_duplicate=True),
              Input("upload", "contents"),
              State("upload", "filename"), prevent_initial_call=True)
def accept_upload(contents, filename):
    if not contents:
        return no_update, no_update
    try:
        return {"ok": True, **parse_tabular_upload(contents, filename)}, None
    except ValueError as exc:
        return ({"ok": False, "error": str(exc), "filename": filename or "uploaded table"},
                None)


@app.callback(Output("loaded-recipe", "data"),
              Output("uploaded-data", "data", allow_duplicate=True),
              Input("recipe-upload", "contents"), State("recipe-upload", "filename"),
              prevent_initial_call=True)
def accept_recipe(contents, filename):
    if not contents:
        return no_update, no_update
    try:
        return {"ok": True, **parse_recipe_upload(contents, filename)}, None
    except (TypeError, ValueError) as exc:
        return ({"ok": False, "error": str(exc),
                 "filename": filename or "recipe JSON"}, None)


def _role_options(columns: list[str], optional: bool = False,
                  none_label: str = "Not used") -> list[dict]:
    options = _options(columns)
    return ([{"label": none_label, "value": NONE_COLUMN}] + options) if optional else options


@app.callback(
    Output("stage", "style"), Output("stage-file", "children"),
    Output("stage-preview", "children"),
    Output("map-subject", "options"), Output("map-subject", "value"),
    Output("map-condition", "options"), Output("map-condition", "value"),
    Output("map-epoch", "options"), Output("map-epoch", "value"),
    Output("map-region", "options"), Output("map-region", "value"),
    Output("map-block", "options"), Output("map-block", "value"),
    Output("map-markers", "options"), Output("map-markers", "value"),
    Output("map-log", "options"), Output("map-log", "value"),
    Output("map-relative", "options"), Output("map-relative", "value"),
    Output("derived-a", "options"), Output("derived-a", "value"),
    Output("derived-inputs", "options"), Output("derived-inputs", "value"),
    Output("epoch-seconds", "value"), Output("derived-name", "value"),
    Output("derived-operation", "value"), Output("recipe-result", "children"),
    Input("dataset", "value"), Input("raw-upload", "data"),
    Input("loaded-recipe", "data"))
def prepare_staging(source, payload, recipe_payload):
    hidden = {"display": "none"}
    empty = (hidden, "", "Choose a table to inspect its columns.",
             [], None, [], None, [], NONE_COLUMN, [], NONE_COLUMN, [], NONE_COLUMN,
             [], [], [], [], [], [], [], None, [], [], EPOCH_SECONDS, None, None, "")
    if source != "upload":
        return empty
    if payload and not payload.get("ok", True):
        return ({}, payload.get("filename", ""), payload.get("error", "invalid upload"),
                [], None, [], None, [], NONE_COLUMN, [], NONE_COLUMN, [], NONE_COLUMN,
                [], [], [], [], [], [], [], None, [], [], EPOCH_SECONDS, None, None,
                "Recipe cannot be applied until a readable table is loaded.")
    frame = raw_uploaded_frame(payload)
    if frame.empty:
        visible = list(empty)
        visible[0] = {}
        if recipe_payload:
            visible[-1] = (recipe_payload.get("error", "Recipe loaded · choose a table to apply it.")
                           if not recipe_payload.get("ok", False)
                           else "Recipe loaded · choose a table to apply it.")
        return tuple(visible)
    defaults = staging_defaults(frame)
    columns = list(frame.columns)
    role_options = _role_options(columns)
    optional_options = _role_options(columns, optional=True)
    epoch_options = _role_options(columns, optional=True, none_label="Use file row order")
    # Options are deliberately broader than the conservative defaults. A recipe can correct a
    # heuristic role guess (for example, a numeric marker whose name contains "id").
    numeric_options = _options(staging_numeric_columns(frame))
    selected = defaults
    recipe_status = ""
    if recipe_payload:
        if not recipe_payload.get("ok", False):
            recipe_status = recipe_payload.get("error", "Invalid recipe JSON.")
        else:
            try:
                settings = validate_recipe(recipe_payload["recipe"], columns=columns)
                selected = {
                    "subject": settings["subject_col"],
                    "condition": settings["condition_col"],
                    "epoch": settings["epoch_col"],
                    "region": settings["region_col"],
                    "block": settings["block_col"],
                    "markers": settings["marker_cols"],
                    "log": settings["log_cols"],
                    "relative": settings["relative_cols"],
                    "epoch_seconds": settings["epoch_seconds"],
                    "derived_name": settings["derived_name"],
                    "derived_operation": settings["derived_operation"],
                    "derived_a": settings["derived_a"],
                    "derived_inputs": settings["derived_inputs"],
                }
                recipe_status = "Recipe loaded · review the mappings, then Stage dataset."
            except (TypeError, ValueError) as exc:
                recipe_status = f"Recipe is incompatible with this table: {exc}"
    return (
        {}, f"{payload.get('filename', 'uploaded table')} · {len(frame):,} rows · "
            f"{len(columns)} columns", _preview_table(frame),
        role_options, selected["subject"], role_options, selected["condition"],
        epoch_options, selected["epoch"] or NONE_COLUMN,
        optional_options, selected["region"] or NONE_COLUMN,
        optional_options, selected["block"] or NONE_COLUMN,
        numeric_options, selected["markers"], numeric_options, selected["log"],
        numeric_options, selected["relative"], numeric_options, selected.get("derived_a"),
        numeric_options, selected.get("derived_inputs", []),
        selected.get("epoch_seconds", EPOCH_SECONDS), selected.get("derived_name"),
        selected.get("derived_operation"), recipe_status,
    )


def staging_settings(subject_col, condition_col, epoch_col, region_col, block_col,
                     marker_cols, log_cols, relative_cols, epoch_seconds,
                     derived_name, derived_operation, derived_a, derived_inputs) -> dict:
    """Collect the staging form into the stable recipe settings contract."""
    optional = lambda value: None if value in (None, NONE_COLUMN) else value
    return {
        "subject_col": subject_col, "condition_col": condition_col,
        "epoch_col": optional(epoch_col), "region_col": optional(region_col),
        "block_col": optional(block_col), "marker_cols": marker_cols or [],
        "log_cols": log_cols or [], "relative_cols": relative_cols or [],
        "epoch_seconds": epoch_seconds, "derived_name": derived_name,
        "derived_operation": derived_operation, "derived_a": derived_a,
        "derived_inputs": derived_inputs or [],
    }


@app.callback(
    Output("uploaded-data", "data"), Output("stage-result", "children"),
    Input("stage-button", "n_clicks"),
    State("raw-upload", "data"), State("map-subject", "value"),
    State("map-condition", "value"), State("map-epoch", "value"),
    State("map-region", "value"), State("map-block", "value"),
    State("map-markers", "value"), State("map-log", "value"),
    State("map-relative", "value"), State("epoch-seconds", "value"),
    State("derived-name", "value"), State("derived-operation", "value"),
    State("derived-a", "value"), State("derived-inputs", "value"),
    prevent_initial_call=True)
def commit_staging(n_clicks, raw_payload, subject_col, condition_col, epoch_col,
                   region_col, block_col, marker_cols, log_cols, relative_cols,
                   epoch_seconds, derived_name, derived_operation, derived_a, derived_inputs):
    if not n_clicks or not raw_payload or not raw_payload.get("ok", False):
        return no_update, "Choose a readable table first."
    try:
        frame = raw_uploaded_frame(raw_payload)
        settings = validate_recipe(make_recipe(staging_settings(
            subject_col, condition_col, epoch_col, region_col, block_col,
            marker_cols, log_cols, relative_cols, epoch_seconds, derived_name,
            derived_operation, derived_a, derived_inputs)), columns=list(frame.columns))
        payload = staged_payload(
            frame, raw_payload.get("filename", "uploaded table"),
            **settings)
        payload["recipe"] = make_recipe(settings)
        return payload, staging_summary(uploaded_bundle(payload).epochs)
    except (TypeError, ValueError) as exc:
        return None, str(exc)


@app.callback(
    Output("uploaded-data", "data", allow_duplicate=True),
    Output("stage-result", "children", allow_duplicate=True),
    Input("map-subject", "value"), Input("map-condition", "value"),
    Input("map-epoch", "value"), Input("map-region", "value"), Input("map-block", "value"),
    Input("map-markers", "value"), Input("map-log", "value"),
    Input("map-relative", "value"), Input("epoch-seconds", "value"),
    Input("derived-name", "value"), Input("derived-operation", "value"),
    Input("derived-a", "value"), Input("derived-inputs", "value"),
    prevent_initial_call=True)
def invalidate_staged_analysis(*_settings):
    return None, "Settings changed · Stage dataset to refresh the analysis."


@app.callback(Output("download-recipe-button", "disabled"),
              Input("dataset", "value"), Input("uploaded-data", "data"))
def disable_recipe_download(source, payload):
    return not (source == "upload" and payload and payload.get("ok", False)
                and payload.get("recipe"))


@app.callback(Output("recipe-download", "data"),
              Input("download-recipe-button", "n_clicks"),
              State("uploaded-data", "data"), prevent_initial_call=True)
def download_recipe(n_clicks, payload):
    if not n_clicks or not payload or not payload.get("recipe"):
        return no_update
    stem = os.path.splitext(os.path.basename(payload.get("filename", "dataset")))[0]
    safe_stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip("-") or "dataset"
    return dcc.send_string(recipe_json(payload["recipe"]),
                           f"{safe_stem}-staging-recipe.json")


@app.callback(Output("analysis-workspace", "style"),
              Input("dataset", "value"), Input("uploaded-data", "data"))
def show_analysis(source, payload):
    if source == "upload" and not (payload and payload.get("ok", False)):
        return {"display": "none"}
    return {}


@app.callback(
    Output("condition", "options"), Output("condition", "value"),
    Output("reference", "options"), Output("reference", "value"),
    Output("marker", "options"), Output("marker", "value"),
    Output("region", "options"), Output("region", "value"), Output("source-status", "children"),
    Input("dataset", "value"), Input("uploaded-data", "data"))
def configure(source, payload):
    if source == "upload" and payload and not payload.get("ok", True):
        return [], None, [], None, [], None, [], None, payload.get("error", "invalid upload")
    bundle = get_bundle(source, payload)
    config = configuration(bundle)
    if not len(bundle.epochs):
        status = ("Choose a table, map its columns, then stage the dataset."
                  if source == "upload" else "Dataset files not found.")
    else:
        status = (f"{bundle.label} · {bundle.epochs['subject'].nunique()} subjects · "
                  f"{bundle.epochs['condition'].nunique()} conditions · "
                  f"{len(markers_available(bundle.epochs))} markers")
    return (*config, status)


@app.callback(
    Output("audit", "children"), Output("g-overview", "figure"),
    Output("g-effects", "figure"), Output("g-floor", "figure"),
    Output("g-reliability", "figure"), Output("g-detectability", "figure"),
    Input("dataset", "value"), Input("uploaded-data", "data"), Input("marker", "value"),
    Input("region", "value"), Input("condition", "value"), Input("reference", "value"),
    Input("target", "value"))
def update(source, payload, marker, region, condition, reference, target):
    bundle = get_bundle(source, payload)
    if not all((len(bundle.epochs), marker, region, condition, reference)):
        blank = _blank("Select a valid dataset and configuration.")
        return [], blank, blank, blank, blank, blank
    measured = measurement(bundle, marker, region, condition, reference, target)
    screened = effect_table(bundle, region, condition, reference)
    audit = [html.Div(className="audit-cell", children=[
        html.Div(key, className="audit-key"), html.Div(value, className="audit-value")])
        for key, value in audit_readout(bundle, marker, region, condition, reference,
                                        target, measured)]
    return (
        audit, fig_condition_overview(bundle, marker, region, reference),
        fig_effects(bundle, marker, region, condition, reference, screened),
        fig_floor(bundle, marker, region, condition, reference, measured),
        fig_reliability(bundle, marker, region, condition, reference, measured),
        fig_detectability(bundle, marker, region, condition, reference, target, measured),
    )


def _free_port(start: int) -> int:
    import socket
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def main():
    port = _free_port(int(os.environ.get("PORT", 8083)))
    print(f"general feasibility explorer -> http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
