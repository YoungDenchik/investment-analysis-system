import pandas as pd
from datetime import timezone
from zoneinfo import ZoneInfo
import types

from data.providers.yfinance_market_provider import YFinanceMarketProvider

def _build_fake_ticker(df, divs=None):
    fake = types.SimpleNamespace()
    fake.history = lambda **__: df.copy()
    fake.dividends = divs or pd.Series(dtype=float)
    return fake

def test_history_success(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=3, freq="D", tz=timezone.utc)
    df = pd.DataFrame(
        {"Open":[1,2,3], "High":[1,2,3],
         "Low":[1,2,3], "Close":[1,2,3], "Volume":[10,10,10]}, index=idx)
    monkeypatch.setattr("yfinance.Ticker", lambda *_: _build_fake_ticker(df))

    provider = YFinanceMarketProvider()
    out = provider.fetch_price_data(
        "AAPL", start=idx[0], end=idx[-1], interval="1d", tz="Europe/Kyiv")
    assert out.index.tz == ZoneInfo("Europe/Kyiv")
    assert list(out.columns) == ["open","high","low","close","volume"]

