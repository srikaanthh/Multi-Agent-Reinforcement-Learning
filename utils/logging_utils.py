from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class EpisodeLogger:
    """Simple in-memory logger for reproducible environment traces."""

    records: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    def dump_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
