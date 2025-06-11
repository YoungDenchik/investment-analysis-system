"""
branches.py
===========

• **ArimaBranch** – робить різницювання/лог-доходи → ARIMA бачить стаціонарний сигнал
• **TreeBranch**  – лишає «як є» (деревам не страшний масштаб); опційно заповнює NA
• **LstmBranch**  – Min-Max 0-1 + перетворює у 3-D (samples, seq_len, features)

Усі класи sklearn-сумісні: fit / transform, тому їх можна помістити всередину
Pipeline й логувати в MLflow.
"""

from __future__ import annotations
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


# ------------------------------------------------------------------ #
# 1.  ARIMA-BRANCH                                                   #
# ------------------------------------------------------------------ #
class ArimaBranch(BaseEstimator, TransformerMixin):
    """
    Перетворює ряд у стаціонарний для ARIMA/SARIMA.

    Parameters
    ----------
    diff_order : int
        Порядок різницювання.
    use_log : bool
        Якщо True — бере log(price) перед diff (≈ log-returns).
    target_col : str
        Колонка, по якій будуємо ряд (звичайно 'close').
    drop_na : bool
        Видалити початкові NaN після diff.
    """
    def __init__(
        self,
        diff_order: int = 1,
        use_log: bool = False,
        target_col: str = "close",
        drop_na: bool = True,
    ):
        self.d = diff_order
        self.use_log = use_log
        self.col = target_col
        self.drop_na = drop_na

    # stateless
    def fit(self, X, y=None): return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        s = X[self.col].astype("float64")
        if self.use_log:
            s = np.log(s.replace(0, np.nan))
        s_diff = s.diff(self.d)
        if self.drop_na:
            s_diff = s_diff.dropna()
        return s_diff.to_frame(f"{self.col}_diff{self.d}")


# ------------------------------------------------------------------ #
# 2.  TREE-BRANCH (RF, XGB, LightGBM)                                #
# ------------------------------------------------------------------ #
class TreeBranch(BaseEstimator, TransformerMixin):
    """
    Дерева нечутливі до масштабу, тому просто передаємо numeric-фічі.

    Parameters
    ----------
    fillna : Literal["ffill","bfill","mean", None]
        Спосіб заповнення NaN перед подачею в модель.
    """
    def __init__(self, fillna: Literal["ffill", "bfill", "mean", None] = None,
                 keep_na: bool = False):

        self.fillna = fillna
        self.keep_na: bool = keep_na


    def fit(self, X, y=None): return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        if not self.keep_na:
            X = X.dropna(how='any')

        return X


# ------------------------------------------------------------------ #
# 3.  LSTM-BRANCH                                                    #
# ------------------------------------------------------------------ #
class LstmBranch(BaseEstimator, TransformerMixin):
    """
    Перетворює DataFrame у 3-D тензор для послідовної мережі.

    Parameters
    ----------
    seq_len : int
        Довжина вікна (кількість попередніх днів).
    scaler : {"minmax","standard",None}
        Який скейлер використати.  Поки тільки Min-Max для простоти.
    keep_na : bool
        Чи залишати зразки, де є NaN після скейлу (False — видаляти).
    dtype : type
        Тип тензора для PyTorch / TF (float32 – оптимально).
    """
    def __init__(
        self,
        seq_len: int = 60,
        scaler: Literal["minmax", None] = "minmax",
        keep_na: bool = False,
        dtype=np.float32,
    ):
        self.seq_len = seq_len
        self.scaler = scaler

        # self.scaler_type = scaler
        self.keep_na = keep_na
        self.dtype = dtype
        self._scaler = MinMaxScaler() if scaler == "minmax" else None

    def fit(self, X, y=None):
        if self._scaler is not None:
            self._scaler.fit(X.select_dtypes("number"))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df_num = X.select_dtypes("number")
        if self._scaler is not None:
            df_num = pd.DataFrame(
                self._scaler.transform(df_num),
                index=df_num.index,
                columns=df_num.columns,
            )
        # 3-D послідовності
        arr = df_num.to_numpy(dtype=self.dtype, copy=False)
        if len(arr) < self.seq_len:
            raise ValueError("Not enough rows for the requested seq_len")
        seq = sliding_window_view(arr, (self.seq_len, arr.shape[1]))[:-1, 0]
        # drop NaN-containing windows?
        if not self.keep_na:
            mask = ~np.isnan(seq).any(axis=(1, 2))
            seq = seq[mask]
        return seq
