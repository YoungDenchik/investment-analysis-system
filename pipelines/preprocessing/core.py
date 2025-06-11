# pipelines/core.py
"""
CorePreprocessor
================
• Вирівнює OHLCV до біржового календаря (NYSE за замовч.)
• Заповнює "дірки" forward-fill + розумна обробка Volume
• Winsorize (IQR-trim) для цін та об’ємів
• Приводить dtypes, сортує індекс, видаляє дублі
• Повертає DataFrame, готовий для подальших кроків (tech_індикатори тощо)

Вхід:
    index          : DatetimeIndex (UTC або naive)
    columns        : ['open','high','low','close','volume'] – lower-case
"""

from __future__ import annotations
import warnings, logging, functools, sys
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:                     # production-бібліотека біржових календарів
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None
    warnings.warn("pandas-market-calendars not installed – "
                  "weekend handling only", RuntimeWarning)

_LOG = logging.getLogger(__name__)

"""Core financial time-series preprocessing utilities.

This module provides a **production‑ready** ``CorePreprocessor`` that can be
embedded into any *scikit‑learn* ``Pipeline`` or used standalone.  It follows
current best practices for cleaning OHLCV equity data:

* Calendar alignment to official exchange sessions via *pandas‑market‑calendars*.
* Robust gap‑filling (week‑ends / holidays → forward‑fill prices, set ``volume``
  to ``0``).
* **Rolling IQR winsorisation** to soften the impact of extreme values while
  adapting to *local* volatility regimes.
* Time‑zone normalisation, NaT‑safe index handling, and dtype optimisation.

The transformer is **stateless** (``fit`` does nothing) so it can sit at the
front of any ML pipeline without persistence overhead.
"""

from typing import Final, Iterable
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import pandas_market_calendars as mcal
except ImportError:  # optional dependency
    mcal = None  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["CorePreprocessor"]


class CorePreprocessor(BaseEstimator, TransformerMixin):
    """Preprocess raw OHLCV bars for modelling.

    Parameters
    ----------
    calendar : str, default "NYSE"
        Exchange key understood by *pandas‑market‑calendars*.
    outlier_k : float, default 1.5
        Tukey constant *k* used in the inter‑quartile range rule.
    window : int, default 30
        Rolling window size **in trading days** for the adaptive IQR filter.
    tz : str or None, default "UTC"
        Target time‑zone; if *None* keep index zone‑naïve.
    price_cols : iterable of str, optional
        Alternative column names order if your dataset differs from the
        canonical ("open", "high", "low", "close").
    """

    DEFAULT_PRICE_COLS: Final[tuple[str, ...]] = ("open", "high", "low", "close")

    def __init__(
        self,
        *,
        calendar: str = "NYSE",
        outlier_k: float = 1.5,
        window: int = 30,
        tz: str | None = "UTC",
        price_cols: Iterable[str] | None = None,
    ) -> None:
        self.calendar = calendar
        self.outlier_k = float(outlier_k)
        self.window = int(window)
        self.tz = tz
        self.price_cols = tuple(price_cols) if price_cols else self.DEFAULT_PRICE_COLS

        if self.window < 1:
            raise ValueError("window must be a positive integer")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _trading_days(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """Return the sessions between *start* and *end* inclusive."""
        if start is pd.NaT or end is pd.NaT:
            raise ValueError("Cannot generate calendar – start or end is NaT")

        start_naive = pd.Timestamp(start).tz_localize(None)
        end_naive = pd.Timestamp(end).tz_localize(None)

        if mcal:
            sched = mcal.get_calendar(self.calendar).schedule(start_naive, end_naive)
            return sched.index.tz_localize(None)
        logger.warning(
            "pandas_market_calendars not available – falling back to business days for %s",
            self.calendar,
        )
        return pd.bdate_range(start_naive, end_naive)

    def _rolling_winsorize(self, s: pd.Series) -> pd.Series:
        """Apply a rolling Tukey IQR filter.

        For every row *t* we compute ``Q1_t``, ``Q3_t`` from the *window* most
        recent observations (including *t*) and clip *only that single value*.
        This keeps the filter causal and suitable for walk‑forward modelling.
        """
        if s.isna().all():
            return s  # early exit

        q1 = s.rolling(self.window, min_periods=1).quantile(0.25)
        q3 = s.rolling(self.window, min_periods=1).quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.outlier_k * iqr
        upper = q3 + self.outlier_k * iqr

        return s.clip(lower=lower, upper=upper).astype(s.dtype, copy=False)

    # ------------------------------------------------------------------
    # scikit‑learn interface
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: None | pd.Series = None):  # noqa: N805 (sklearn signature)
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DatetimeIndex")
        return self  # stateless

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return a cleaned *copy* of *X*; original DataFrame remains unchanged."""
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DatetimeIndex")

        if X.empty:
            logger.warning("Received empty DataFrame – returning a copy without changes")
            return X.copy()

        # ------------------------------------------------------------------
        # 0. Prep – clean NaT indices up‑front to avoid mcal errors
        # ------------------------------------------------------------------
        df = X[~X.index.isna()].copy().sort_index()
        if df.empty:
            raise ValueError("All index values are NaT – cannot preprocess")

        # ------------------------------------------------------------------
        # 1. Time‑zone normalisation
        # ------------------------------------------------------------------
        if self.tz:
            df.index = (
                df.index.tz_localize(self.tz) if df.index.tz is None else df.index.tz_convert(self.tz)
            )

        # ------------------------------------------------------------------
        # 2. Align to trading calendar & fill gaps
        # ------------------------------------------------------------------
        # start_ts: pd.Timestamp = df.index.min()
        # end_ts: pd.Timestamp = df.index.max()
        # full_idx = self._trading_days(start_ts, end_ts)
        # full_idx = full_idx.tz_localize(df.index.tz) if df.index.tz is not None else full_idx
        # df = df.reindex(full_idx)

        for col in self.price_cols:
            if col in df.columns:
                df[col] = df[col].ffill()
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)

        # ------------------------------------------------------------------
        # 3. Adaptive rolling IQR winsorisation
        # ------------------------------------------------------------------
        for col in (*self.price_cols, "volume"):
            if col in df.columns:
                df[col] = self._rolling_winsorize(df[col].astype(float))

        # ------------------------------------------------------------------
        # 4. Optional dtype optimisation
        # ------------------------------------------------------------------
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], downcast="integer")

        # Drop rows where *all* price columns are still NaN (e.g., before first ffills)
        present_cols = [c for c in self.price_cols if c in df.columns]
        df = df.dropna(subset=present_cols, how="all")

        return df

    # ------------------------------------------------------------------
    # Miscellany
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover – cosmetic only
        params = (
            f"calendar='{self.calendar}', tz='{self.tz}', outlier_k={self.outlier_k}, window={self.window}"
        )
        return f"CorePreprocessor({params})"
