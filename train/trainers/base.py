from __future__ import annotations

from typing import Protocol, Tuple, Any

import pandas as pd

from pipelines.preprocessing.manager import FeaturePipelineManager


class BaseTrainer(Protocol):
    """Minimal public interface each trainer must implement."""

    def preprocess(self, df: pd.DataFrame) -> Tuple[Any, Any]: ...

    def fit(self, X, y): ...

    def predict(self, X): ...


class TabularTrainerMixin:
    """Shared helper applying (cached) feature pipeline."""

    def apply_pipeline(self, X: pd.DataFrame, model_tag: str, cfg_path):
        pipe = FeaturePipelineManager.get(model_tag, cfg_path)
        return pipe.transform(X) if pipe else X

