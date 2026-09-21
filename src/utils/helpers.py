"""Configuration, paths, JSON persistence, and domain errors."""

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
RISK_CLASSES = ("LOW", "MEDIUM", "HIGH")

class DataValidationError(ValueError):
    """A dataset cannot safely be used in the pipeline."""

class DatasetNotFoundError(FileNotFoundError):
    """Configured raw data are absent."""

class ModelNotFoundError(FileNotFoundError):
    """Train a model before requesting inference."""

def resolve_path(path: str | Path) -> Path:
    """Resolve relative configuration paths against the repository root."""
    value = Path(path)
    return value if value.is_absolute() else ROOT / value

def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping, failing clearly on malformed configuration."""
    with resolve_path(path).open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return result

def save_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write readable strict JSON; never serialize NaN as a numeric result."""
    destination = resolve_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")

def read_json(path: str | Path) -> dict[str, Any]:
    """Read persisted metadata."""
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))
