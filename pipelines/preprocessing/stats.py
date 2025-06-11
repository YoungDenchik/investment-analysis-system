"""
RollingStats
============

✓ Додає до DataFrame набір ковзних статистик за один прохід.
✓ Параметризується:
      • windows  – список або кортеж довжин вікна
      • stats    – які саме метрики рахувати
      • price_col, vol_col   – з якої серії брати значення
✓ Підтримує annualised volatility (`vol_σ * √252`) і Z-score.

Повертає той самий DataFrame + нові колонки:
    close_mean20, close_std20, close_z20, vol_60d, …

Додайте трансформер в `_REGISTRY` build-файла:
    "rolling_stats": "pipelines.stats:RollingStats"
"""

from __future__ import annotations
from typing import Iterable, Literal
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RollingStats(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    windows : Iterable[int]
        Список/кортеж ковзних вікон (днів).  Example: (5, 20, 60)
    stats : Iterable[str]
        Обираємо з {'mean','std','min','max','skew','kurt','zscore','vol'}
        • 'std'   – rolling σ (не annualised)
        • 'vol'   – annualised σ = std * √ trading_days
    price_col : str
        Cерію, над якою виводимо статистики (`close` за замовч.)
    trading_days : int
        Для annualised volatility (vol)  [252 або 365]
    drop_na : bool
        Видалити рядки, де всі нові фічі NaN (часто на початку ряду)
    """

    _ALLOWED = {"mean", "std", "min", "max", "skew", "kurt", "zscore", "vol"}

    def __init__(
        self,
        windows: Iterable[int] = (5, 20, 60),
        stats: Iterable[str] = ("mean", "std", "zscore", "vol"),
        price_col: str = "close",
        trading_days: int = 252,
        drop_na: bool = False,
    ):
        self.windows = tuple(sorted(set(int(w) for w in windows)))
        self.stats = tuple(stats)
        self.price_col = price_col
        self.trading_days = trading_days
        self.drop_na = drop_na

        unknown = set(self.stats) - self._ALLOWED
        if unknown:
            raise ValueError(f"Unknown stats {unknown}")

    # stateless
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.price_col not in X.columns:
            raise KeyError(f"Column '{self.price_col}' missing from DataFrame")

        df = X.copy()
        s = df[self.price_col].astype("float64")

        for w in self.windows:
            roll = s.rolling(window=w, min_periods=w)

            if "mean" in self.stats:
                df[f"{self.price_col}_mean{w}"] = roll.mean()

            if "std" in self.stats or "vol" in self.stats or "zscore" in self.stats:
                rstd = roll.std(ddof=0)
                if "std" in self.stats:
                    df[f"{self.price_col}_std{w}"] = rstd
                if "vol" in self.stats:
                    df[f"{self.price_col}_vol{w}"] = rstd * np.sqrt(self.trading_days)
                if "zscore" in self.stats:
                    rmean = roll.mean()
                    df[f"{self.price_col}_z{w}"] = (s - rmean) / rstd

            if "min" in self.stats:
                df[f"{self.price_col}_min{w}"] = roll.min()
            if "max" in self.stats:
                df[f"{self.price_col}_max{w}"] = roll.max()
            if "skew" in self.stats:
                df[f"{self.price_col}_skew{w}"] = roll.skew()
            if "kurt" in self.stats:
                df[f"{self.price_col}_kurt{w}"] = roll.kurt()

        # if self.drop_na:
        #     new_cols = [c for c in df.columns if re_search := c.startswith(self.price_col) and any(str(w) in c for w in self.windows)]
        #     df = df.dropna(subset=new_cols)

        return df

