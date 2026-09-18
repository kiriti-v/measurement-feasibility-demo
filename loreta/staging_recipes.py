"""Versioned, data-free recipes for staging repeated-measures tables.

Recipes deliberately contain column mappings and transformations only.  They never
contain uploaded rows, filenames, paths, or subject identifiers from those rows.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = 1

_SETTINGS_FIELDS = {
    "subject_col",
    "condition_col",
    "epoch_col",
    "region_col",
    "block_col",
    "marker_cols",
    "log_cols",
    "relative_cols",
    "epoch_seconds",
    "derived_name",
    "derived_operation",
    "derived_a",
    "derived_inputs",
}
_ROLE_FIELDS = ("subject_col", "condition_col", "epoch_col", "region_col", "block_col")
_LIST_FIELDS = ("marker_cols", "log_cols", "relative_cols", "derived_inputs")
_DERIVED_OPERATIONS = {"log10", "ratio", "difference", "share"}
_CANONICAL_ROLE_NAMES = {
    "subject", "condition", "epoch", "region", "block", "state", "t_start",
    "n_channels", "group", "is_baseline",
}
_ROLE_OUTPUTS = {
    "subject_col": "subject",
    "condition_col": "condition",
    "epoch_col": "epoch",
    "region_col": "region",
    "block_col": "block",
}


def _optional_string(value: Any, field: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"settings.{field} must be a string or null")
    if not value.strip():
        raise ValueError(f"settings.{field} cannot be blank")
    return value


def _required_string(value: Any, field: str) -> str:
    normalized = _optional_string(value, field)
    if normalized is None:
        raise ValueError(f"settings.{field} is required")
    return normalized


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"settings.{field} must be a list of column-name strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"settings.{field} must contain only non-blank strings")
        if item not in result:
            result.append(item)
    return result


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "derived_marker"


def _normalized_settings(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, Mapping):
        raise ValueError("recipe settings must be an object")

    unknown = set(settings) - _SETTINGS_FIELDS
    if unknown:
        raise ValueError("unknown recipe setting(s): " + ", ".join(sorted(map(str, unknown))))
    non_string_keys = [key for key in settings if not isinstance(key, str)]
    if non_string_keys:
        raise ValueError("recipe setting names must be strings")

    normalized: dict[str, Any] = {
        "subject_col": _required_string(settings.get("subject_col"), "subject_col"),
        "condition_col": _required_string(settings.get("condition_col"), "condition_col"),
        "epoch_col": _optional_string(settings.get("epoch_col"), "epoch_col"),
        "region_col": _optional_string(settings.get("region_col"), "region_col"),
        "block_col": _optional_string(settings.get("block_col"), "block_col"),
        "marker_cols": _string_list(settings.get("marker_cols", []), "marker_cols"),
        "log_cols": _string_list(settings.get("log_cols", []), "log_cols"),
        "relative_cols": _string_list(settings.get("relative_cols", []), "relative_cols"),
        "derived_name": _optional_string(settings.get("derived_name"), "derived_name"),
        "derived_operation": _optional_string(
            settings.get("derived_operation"), "derived_operation"
        ),
        "derived_a": _optional_string(settings.get("derived_a"), "derived_a"),
        "derived_inputs": _string_list(settings.get("derived_inputs", []), "derived_inputs"),
    }

    if "epoch_seconds" not in settings:
        raise ValueError("settings.epoch_seconds is required")
    duration = settings["epoch_seconds"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError("settings.epoch_seconds must be a finite number greater than zero")
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("settings.epoch_seconds must be a finite number greater than zero")
    normalized["epoch_seconds"] = duration

    roles = [normalized[field] for field in _ROLE_FIELDS if normalized[field] is not None]
    if len(roles) != len(set(roles)):
        raise ValueError("each dataset role must use a different column")
    if not normalized["marker_cols"]:
        raise ValueError("settings.marker_cols must select at least one measurement column")
    role_overlap = set(roles) & set(normalized["marker_cols"])
    if role_overlap:
        raise ValueError(
            "role columns cannot also be measurements: " + ", ".join(sorted(role_overlap))
        )
    markers = set(normalized["marker_cols"])
    reserved_markers = markers & _CANONICAL_ROLE_NAMES
    if reserved_markers:
        raise ValueError(
            "measurement names are reserved for canonical roles: "
            + ", ".join(sorted(reserved_markers))
        )
    output_names = [
        _ROLE_OUTPUTS[field] for field in _ROLE_FIELDS if normalized[field] is not None
    ] + normalized["marker_cols"]
    if len(output_names) != len(set(output_names)):
        raise ValueError("mapped column names collide after applying canonical role names")

    for scale_field in ("log_cols", "relative_cols"):
        outside = set(normalized[scale_field]) - markers
        if outside:
            raise ValueError(
                f"settings.{scale_field} must be a subset of marker_cols; not selected: "
                + ", ".join(sorted(outside))
            )
    scale_overlap = set(normalized["log_cols"]) & set(normalized["relative_cols"])
    if scale_overlap:
        raise ValueError(
            "measurements cannot be both log-scaled and relative: "
            + ", ".join(sorted(scale_overlap))
        )

    operation = normalized["derived_operation"]
    derived_a = normalized["derived_a"]
    derived_inputs = normalized["derived_inputs"]
    if operation is None:
        if normalized["derived_name"] is not None or derived_a is not None or derived_inputs:
            raise ValueError(
                "derived_name, derived_a, and derived_inputs require derived_operation"
            )
    elif operation not in _DERIVED_OPERATIONS:
        allowed = ", ".join(sorted(_DERIVED_OPERATIONS))
        raise ValueError(f"unknown derived operation {operation!r}; expected one of: {allowed}")
    elif derived_a is None:
        raise ValueError("settings.derived_a is required for a derived operation")
    elif operation == "log10" and derived_inputs:
        raise ValueError("log10 requires no comparison inputs")
    elif operation in {"ratio", "difference"} and len(derived_inputs) != 1:
        raise ValueError(f"{operation} requires exactly one comparison input")
    elif operation == "share" and not derived_inputs:
        raise ValueError("share requires at least one denominator input")

    if operation is not None:
        operand_roles = ({derived_a} | set(derived_inputs)) & set(roles)
        if operand_roles:
            raise ValueError(
                "dataset role columns cannot be derived-marker inputs: "
                + ", ".join(sorted(operand_roles))
            )
        derived_output = _slug(normalized["derived_name"] or operation)
        if derived_output in _CANONICAL_ROLE_NAMES or derived_output in output_names:
            raise ValueError(f"derived marker name {derived_output!r} is already in use or reserved")

    return normalized


def validate_recipe(recipe: dict, columns: list[str] | None = None) -> dict:
    """Validate *recipe* and return its normalized ``settings`` mapping."""
    if not isinstance(recipe, Mapping):
        raise ValueError("recipe must be a JSON object")
    unknown = set(recipe) - {"schema_version", "settings"}
    if unknown:
        raise ValueError("unknown top-level recipe field(s): " + ", ".join(sorted(map(str, unknown))))
    if set(recipe) != {"schema_version", "settings"}:
        missing = {"schema_version", "settings"} - set(recipe)
        raise ValueError("recipe is missing required field(s): " + ", ".join(sorted(missing)))

    version = recipe["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("schema_version must be the integer 1")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported recipe schema_version {version}; expected {SCHEMA_VERSION}"
        )

    settings = _normalized_settings(recipe["settings"])
    if columns is not None:
        if not isinstance(columns, list) or any(not isinstance(column, str) for column in columns):
            raise ValueError("columns must be a list of strings")
        available = set(columns)
        referenced = {
            settings[field] for field in _ROLE_FIELDS if settings[field] is not None
        }
        referenced.update(settings["marker_cols"])
        if settings["derived_a"] is not None:
            referenced.add(settings["derived_a"])
        referenced.update(settings["derived_inputs"])
        missing = referenced - available
        if missing:
            raise ValueError("recipe columns not found in table: " + ", ".join(sorted(missing)))
    return settings


def make_recipe(settings: dict) -> dict:
    """Build a canonical schema-v1 recipe from staging settings."""
    recipe = {"schema_version": SCHEMA_VERSION, "settings": settings}
    return {"schema_version": SCHEMA_VERSION, "settings": validate_recipe(recipe)}


def recipe_json(recipe: dict) -> str:
    """Serialize a validated recipe as compact, deterministic JSON."""
    canonical = make_recipe(validate_recipe(recipe))
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parse_recipe(contents: str) -> dict:
    """Parse raw JSON text and return a canonical validated recipe."""
    if not isinstance(contents, str):
        raise ValueError("recipe contents must be a JSON string")
    try:
        parsed = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ValueError(f"recipe is not valid JSON: {exc.msg}") from exc
    settings = validate_recipe(parsed)
    return {"schema_version": SCHEMA_VERSION, "settings": settings}
