from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from pipelines.preprocessing.build import build as _build_pipeline


class FeaturePipelineManager:
    """Cache + factory for feature pipelines (singleton‑style)."""

    _cache: Dict[Tuple[str, str | None], object] = {}

    @classmethod
    def get(cls, model_tag: str, cfg_path: str | Path | None):
        key = (model_tag, str(cfg_path))
        if key not in cls._cache:
            cls._cache[key] = _build_pipeline(model_tag, cfg_path) if cfg_path else None
        return cls._cache[key]
