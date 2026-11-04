"""A data folder's `goose-plotter.json`: labels, units, sample colours, default axes,
format and step recipes."""

import json
from dataclasses import dataclass, field
from pathlib import Path

PROFILE_NAME = "goose-plotter.json"


@dataclass
class Profile:
    """Everything optional; an empty profile shows raw column names."""
    labels: dict = field(default_factory=dict)  # column -> axis label
    units: dict = field(default_factory=dict)  # column -> unit
    samples: dict = field(default_factory=dict)  # sample name -> colour
    defaults: dict = field(default_factory=dict)  # 'x'/'y' -> column
    format: dict | None = None  # datasets.load_dataset format; None: detect
    # name -> {"x", "x_fn": the x it was made on, "steps": as session.dump_steps}
    recipes: dict = field(default_factory=dict)


def _read(data_dir):
    path = Path(data_dir) / PROFILE_NAME
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")  # noqa: TRY004 - callers catch ValueError
    return data


def load_profile(data_dir):
    """(profile, error): the folder's profile, or an empty one and why it failed."""
    if not data_dir:
        return Profile(), ""
    try:
        data = _read(data_dir)
        kinds = {"labels": dict, "units": dict, "samples": dict, "defaults": dict,
                 "format": (dict, type(None)), "recipes": dict}
        for key, kind in kinds.items():
            if not isinstance(data.get(key), kind) and key in data:
                raise ValueError(f"'{key}' has the wrong type")
        profile = Profile(**{k: data[k] for k in kinds if k in data})
        profile.recipes = {name: r for name, r in profile.recipes.items()
                           if isinstance(r, dict) and isinstance(r.get("steps"), list)}
    except (OSError, ValueError) as err:  # json errors are ValueErrors
        return Profile(), f"{PROFILE_NAME} ignored: {err}"
    if profile.format is not None and not _valid_format(profile.format):
        profile.format = None  # keep the rest of the profile
        return profile, f"{PROFILE_NAME}: 'format' ignored, see README"
    return profile, ""


def _valid_format(fmt):
    def whole(value, least):
        return type(value) is int and value >= least
    return (isinstance(fmt.get("delimiter"), str) and fmt["delimiter"] != ""
            and whole(fmt.get("header_line"), 0) and whole(fmt.get("data_line"), 1))


def _save(data_dir, key, value):
    """Store `value` (None removes it) under `key` in the folder's profile,
    keeping the rest."""
    data = _read(data_dir)
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    (Path(data_dir) / PROFILE_NAME).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def save_format(data_dir, fmt):
    """Store `fmt` (None removes it) in the folder's profile, keeping the rest."""
    _save(data_dir, "format", fmt)


def save_recipes(data_dir, recipes):
    """Store every recipe in the folder's profile (none removes the key), keeping the rest."""
    _save(data_dir, "recipes", recipes or None)
