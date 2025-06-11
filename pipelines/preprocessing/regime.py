"""
RegimeDetector
==============

Визначає **bull / bear / sideways** режими за 2-хвильовою логікою
(коротка SMA vs довга SMA) або за кумулятивним %-зміною.

Додає до DataFrame:

    •  regime         : {-1, 0, 1}   (-1=Bear, 0=Sideways, 1=Bull)
    •  is_bull / is_bear / is_side  (dtype uint8) — one-hot (опційно)

Чому важливо:
    • моделі дерев / XGB часто користуються цим як “мікрошаблоном” тренду;
    • можна відсікати періоди flat-range у бектестах.

"""

from __future__ import annotations
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RegimeDetector(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    short_win : int
        Коротке вікно SMA.   Δ(ціна) > 0 («mini-trend»).
    long_win : int
        Довге вікно SMA.     Інструмент повинен перетинати саме його.
    pct_band : float
        Якщо |SMA_short − SMA_long| / SMA_long < pct_band → sideways.
    one_hot : bool
        Додати окремі стовпці is_bull / is_bear / is_side.
    price_col : str
        Назва цінової колонки (звичайно 'close').
    drop_na : bool
        Видаляємо рядки до long_win? (True у train-час, False у online).
    """

    def __init__(
        self,
        short_win: int = 20,
        long_win: int = 60,
        pct_band: float = 0.01,
        one_hot: bool = True,
        price_col: str = "close",
        drop_na: bool = False,
    ):
        if short_win >= long_win:
            raise ValueError("short_win must be < long_win")
        self.s = short_win
        self.l = long_win
        self.band = pct_band
        self.one_hot = one_hot
        self.col = price_col
        self.drop_na = drop_na

    # stateless
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.col not in X.columns:
            raise KeyError(f"Column '{self.col}' not found.")

        df = X.copy()
        s = df[self.col].astype("float64")

        sma_s = s.rolling(self.s, min_periods=self.s).mean()
        sma_l = s.rolling(self.l, min_periods=self.l).mean()

        # delta від довгої SMA
        rel = (sma_s - sma_l) / sma_l

        regime = np.select(
            [
                rel > self.band,              # bull
                rel < -self.band              # bear
            ],
            [1, -1],
            default=0                        # sideways
        )

        df["regime"] = regime.astype("int8")

        if self.one_hot:
            df["is_bull"]  = (regime == 1).astype("uint8")
            df["is_bear"]  = (regime == -1).astype("uint8")
            df["is_side"]  = (regime == 0).astype("uint8")

        if self.drop_na:
            df = df.iloc[self.l :]           # щоб не було NaN у SMA

        return df
