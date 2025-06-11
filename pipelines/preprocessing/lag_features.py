from __future__ import annotations
from typing import Iterable, Literal
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


"""
LagFeatures
===========

✓ Створює *авторегресивні* фічі з довільним набором лагів.

Параметри
---------
lags : Iterable[int]
    Список лагів у торгових днях.     → [1,5,10]
columns : list[str] | None
    Які стовпці лагувати.
    None ⇒ ['close'].
kind : {'raw','diff','pct'}
    • raw  – значення t-lag           (Close_lag5)  
    • diff – різниця  (x_t – x_{t-lag})    (Close_diff5)  
    • pct  – відносна зміна (x_t / x_{t-lag} – 1)  (Close_ret5)
drop_na : bool
    True  – видалити перші max(lags) рядків із NaNʼами.
"""


class LagFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        lags: Iterable[int] = (1, 5),
        columns: list[str] | None = None,
        kind: Literal["raw", "diff", "pct"] = "raw",
        drop_na: bool = False,
    ):
        self.lags = tuple(sorted(set(int(l) for l in lags)))
        self.columns = columns        # None ⇒ задамо у transform
        self.kind = kind
        self.drop_na = drop_na

    # stateless
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if self.columns is None:
            self.columns = ["close"]

        for col in self.columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

            for lag in self.lags:
                if self.kind == "raw":
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)
                elif self.kind == "diff":
                    df[f"{col}_diff{lag}"] = df[col] - df[col].shift(lag)
                elif self.kind == "pct":
                    df[f"{col}_ret{lag}"] = df[col].pct_change(lag)
                else:       # теоретично не станеться
                    raise ValueError(f"Unsupported kind '{self.kind}'")

        if self.drop_na:
            df = df.dropna(subset=[c for c in df.columns if any(s in c for s in ("lag", "diff", "ret"))])

        return df.astype("float32", errors="ignore")   # memory-friendly


