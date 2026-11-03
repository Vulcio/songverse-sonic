"""Internal analyzer boundary and the existing v1.6 combination policy.

No service, persistence or inference-library imports belong here. This is the
legacy full-analysis interface, not the future versioned rhythm result schema.
Paths describe a selected input snapshot; callers still own file retention and
alignment. Budgets are advisory until supervised execution is implemented.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

log = logging.getLogger("demucs-api.analysis")


@dataclass(frozen=True)
class AnalysisInput:
    files: tuple[Path, ...]
    source: Path | None
    work_dir: Path
    device: str = "cpu"


class AnalysisBudget(Protocol):
    def ok(self) -> bool: ...


class Analyzer(Protocol):
    """Return raw optional findings, or raise for the caller to handle.

    Legacy backends may include the private `_duration` validation hint.
    Engines must not mutate jobs, deliver webhooks or persist results.
    """

    engine_id: str

    def analyze(self, inputs: AnalysisInput, budget: AnalysisBudget) -> dict[str, Any]: ...


class LegacyAnalysisAdapter:
    """Preserve optional-primary, baseline-fill and sanitization behavior."""

    def __init__(
        self,
        baseline: Analyzer,
        sanitize: Callable[[Any, float | None], dict | None],
        primary: Analyzer | None = None,
    ):
        self.baseline = baseline
        self.primary = primary
        self.sanitize = sanitize

    def analyze(self, inputs: AnalysisInput, budget: AnalysisBudget) -> dict | None:
        raw: dict[str, Any] = {}
        duration = None
        try:
            if self.primary is not None:
                try:
                    raw = self.primary.analyze(inputs, budget)
                except Exception:
                    log.exception("analysis: %s backend failed; falling back to librosa", self.primary.engine_id)
            fallback = self.baseline.analyze(inputs, budget)
            duration = fallback.pop("_duration", None)
            # Preserve the legacy field group exactly during this extraction.
            # Versioned rhythm validation/projection is a separate milestone.
            beat_keys = {"tempo", "first_beat", "beats", "downbeats", "beats_confidence", "intro_free"}
            if "beats" in raw:
                fallback = {k: v for k, v in fallback.items() if k not in beat_keys}
            for k, v in fallback.items():
                if k not in raw:
                    raw[k] = v
        except Exception:
            log.exception("analysis failed")
        return self.sanitize(raw, duration)
