"""Experiment configuration loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML experiment configuration."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise ImportError("Install PyYAML to load YAML experiment configs") from exc
        loaded = yaml.safe_load(text)
        return loaded or {}
    raise ValueError(f"Unsupported config format: {path.suffix}")

