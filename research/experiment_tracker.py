"""Experiment tracking and reproducibility metadata."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRecord:
    """Serializable experiment metadata."""

    experiment_id: str
    name: str
    created_at: str
    config: dict[str, Any]
    metrics: dict[str, float]
    parameters: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: str | None = None


class ExperimentTracker:
    """Persist experiment configs, metrics, artifacts, and comparisons."""

    def __init__(self, root_dir: str | Path = "outputs/experiments") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def start_run(
        self,
        name: str,
        config: dict[str, Any],
        metrics: dict[str, float] | None = None,
        parameters: dict[str, Any] | None = None,
        artifacts: dict[str, str] | None = None,
        notes: str | None = None,
    ) -> ExperimentRecord:
        """Create and persist a timestamped experiment record."""

        experiment_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        record = ExperimentRecord(
            experiment_id=experiment_id,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
            config=config,
            metrics=metrics or {},
            parameters=parameters or {},
            artifacts=artifacts or {},
            notes=notes,
        )
        self.save(record)
        return record

    def save(self, record: ExperimentRecord) -> Path:
        """Persist an experiment record to disk."""

        run_dir = self.root_dir / record.experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "experiment.json"
        path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load(self, experiment_id: str) -> ExperimentRecord:
        """Load an experiment record."""

        path = self.root_dir / experiment_id / "experiment.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return ExperimentRecord(**data)

    def list_runs(self) -> list[ExperimentRecord]:
        """Load all available experiment records."""

        records = []
        for path in sorted(self.root_dir.glob("*/experiment.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append(ExperimentRecord(**data))
        return records

    def compare(self) -> list[dict[str, Any]]:
        """Return flattened records for experiment comparison tables."""

        rows = []
        for record in self.list_runs():
            rows.append(
                {
                    "experiment_id": record.experiment_id,
                    "name": record.name,
                    "created_at": record.created_at,
                    **{f"metric_{key}": value for key, value in record.metrics.items()},
                    **{f"param_{key}": value for key, value in record.parameters.items()},
                }
            )
        return rows

