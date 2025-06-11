from __future__ import annotations
from typing import Iterable, Mapping
import warnings

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import ta  # pip install ta

# ------------------------------------------------------------------ #
# 0.  Default windows                                                #
# ------------------------------------------------------------------ #
_DEFAULTS: dict[str, int | None] = {
    "SMA": 20,
    "EMA": 20,
    # "TEMA": 30,
    "RSI": 14,
    "ATR": 14,
    "ROC": 10,
    "VOL_STD": 60,
    "BBANDS": 20,
    "PPO": None,
    "OBV": None,
    "HLC3": None,
}

# ------------------------------------------------------------------ #
# 1.  Map name  →  builder                                           #
#      Функції отримують (df, win) і повертають pd.DataFrame / pd.Series #
# ------------------------------------------------------------------ #
def _sma(df, w): return ta.trend.SMAIndicator(df["close"], w).sma_indicator()
def _ema(df, w): return ta.trend.EMAIndicator(df["close"], w).ema_indicator()
# def _tema(df, w): return ta.trend.TEMAIndicator(df["close"], w).tema_indicator()
def _rsi(df, w): return ta.momentum.RSIIndicator(df["close"], w).rsi()
def _atr(df, w): return ta.volatility.average_true_range(df["high"], df["low"], df["close"], w)
def _obv(df, _): return ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
def _hlc3(df, _): return (df["high"] + df["low"] + df["close"]) / 3
def _roc(df, w): return ta.momentum.ROCIndicator(df["close"], w).roc()
def _vol(df, w): return df["close"].rolling(w).std() / df["close"]
def _bb(df, w):
    bb = ta.volatility.BollingerBands(df["close"], w)
    return pd.concat([
        bb.bollinger_hband(),
        bb.bollinger_mavg(),
        bb.bollinger_lband()
    ], axis=1).set_axis([f"BBU{w}", f"BBM{w}", f"BBL{w}"], axis=1)
def _ppo(df, _):
    ppo = ta.momentum.PercentagePriceOscillator(df["close"])
    return pd.concat([
        ppo.ppo(),
        ppo.ppo_signal(),
        ppo.ppo_hist()
    ], axis=1).set_axis(["PPO", "PPOsig", "PPOhist"], axis=1)

_FUNC_MAP = {
    "SMA": _sma,
    "EMA": _ema,
    # "TEMA": _tema,
    "RSI": _rsi,
    "ATR": _atr,
    "OBV": _obv,
    "HLC3": _hlc3,
    "ROC": _roc,
    "VOL_STD": _vol,
    "BBANDS": _bb,
    "PPO": _ppo,
}

# ------------------------------------------------------------------ #
# 2.  Трансформер                                                    #
# ------------------------------------------------------------------ #
class TechnicalIndicators(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    indicators : Iterable[str] | Mapping[str, int | None], default=("SMA",
    #"TEMA",
    "ATR")
        • list[str]              – використовує дефолтне вікно зі _DEFAULTS;
        • dict[str, int|None]    – <name>: <window>. None → функції без window.
    drop_na : bool, default=False
        Якщо True → видалити рядки, де усі нові індикатори = NaN
    """

    def __init__(
        self,
        indicators: Iterable[str] | Mapping[str, int | None] = ("SMA",
                                                                #"TEMA",
                                                                "ATR"),
        drop_na: bool = False,
    ):
        self.indicators = indicators
        self.drop_na = drop_na

        # нормалізуємо в dict
        if isinstance(indicators, Mapping):
            self._ind_dict = dict(indicators)
        else:
            self._ind_dict = {name: _DEFAULTS[name] for name in indicators}

        # валідація
        unknown = set(self._ind_dict) - _FUNC_MAP.keys()
        if unknown:
            raise ValueError(f"Unknown indicators: {unknown}")

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        new_cols = []

        for name, win in self._ind_dict.items():
            try:
                series_or_df = _FUNC_MAP[name](df, win)
                if isinstance(series_or_df, pd.Series):
                    col_name = f"{name}{'' if win is None else win}"
                    df[col_name] = series_or_df
                    new_cols.append(col_name)
                else:  # DataFrame
                    df = pd.concat([df, series_or_df], axis=1)
                    new_cols.extend(series_or_df.columns)
            except Exception as e:
                warnings.warn(f"Failed to compute {name}({win}): {e}")
                continue

        if self.drop_na:
            df = df.dropna(subset=new_cols)

        return df
