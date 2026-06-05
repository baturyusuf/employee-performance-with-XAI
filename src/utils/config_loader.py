from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(ValueError):
    """Raised when a project config file cannot be loaded or validated."""


def resolve_config_path(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)

    if path.is_absolute():
        return path

    if path.suffix:
        candidate = CONFIG_DIR / path
    else:
        candidate = CONFIG_DIR / f"{path}.yaml"

    return candidate


def load_config(name_or_path: str | Path) -> Dict[str, Any]:
    """Load a config file.

    Config files are valid YAML. They are intentionally written as JSON-compatible
    YAML so the repository has no hard dependency on PyYAML for phase-1 tests.
    If PyYAML is installed, it is used first; otherwise JSON parsing is used.
    """
    path = resolve_config_path(name_or_path)

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")

    data: Any
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = json.loads(text)
    except Exception as exc:
        raise ConfigError(f"Failed to parse config {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config must contain a mapping/object: {path}")

    return data


def load_project_config() -> Dict[str, Any]:
    return load_config("project")


def load_feature_sets_config() -> Dict[str, Any]:
    data = load_config("feature_sets")
    if "feature_sets" not in data or not isinstance(data["feature_sets"], dict):
        raise ConfigError("feature_sets.yaml must define a 'feature_sets' object.")
    return data


def load_feature_taxonomy_config() -> Dict[str, Any]:
    data = load_config("feature_taxonomy")
    rows = data.get("feature_taxonomy")
    if not isinstance(rows, list):
        raise ConfigError("feature_taxonomy.yaml must define a 'feature_taxonomy' list.")

    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ConfigError("Each feature taxonomy entry must be an object.")
        feature_name = row.get("feature_name")
        if not feature_name:
            raise ConfigError("Each feature taxonomy entry requires feature_name.")
        if feature_name in seen:
            raise ConfigError(f"Duplicate feature taxonomy entry: {feature_name}")
        seen.add(feature_name)

    return data
