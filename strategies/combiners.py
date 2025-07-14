from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

class BaseCombiner:
    """Base class for combiners. Implements optional fit and enforce combine interface."""

    def fit(self, preds: pd.DataFrame, target: pd.Series) -> None:
        """Optional fit method for combiners that require training. Default is no-op."""
        return None

    def combine(self, preds: pd.DataFrame) -> np.ndarray:
        """Combine multiple model predictions into a single array. Must be overridden."""
        raise NotImplementedError("Combine method must be implemented by subclasses.")

class MeanCombiner(BaseCombiner):
    """Combiner that averages predictions across models."""

    def combine(self, preds: pd.DataFrame) -> np.ndarray:
        # Compute the row-wise mean of the predictions
        return preds.mean(axis=1).values

class MedianCombiner(BaseCombiner):
    """Combiner that takes the median of predictions across models."""

    def combine(self, preds: pd.DataFrame) -> np.ndarray:
        # Compute the row-wise median of the predictions
        return preds.median(axis=1).values
