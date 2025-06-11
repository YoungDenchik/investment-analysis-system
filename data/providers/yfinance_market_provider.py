# providers/yahoo_provider.py
from __future__ import annotations

# from data.providers.interfaces.i_market_provider import IMarketProvider

import logging

logger = logging.getLogger(__name__)

import logging
from datetime import datetime, timezone
from typing import Final, Optional

import pandas as pd
import tenacity
import yfinance as yf
from zoneinfo import ZoneInfo

from data.providers.interfaces.i_market_provider import IMarketProvider

_LOG: Final = logging.getLogger(__name__)

_ALLOWED_INTERVALS: Final[set[str]] = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
}

# --------------------------------------------------------------------------- #
# Retry policy (3 tries, expo back-off, log on each failure)                  #
# --------------------------------------------------------------------------- #
retry_policy = tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1.5, min=1.0, max=8.0),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=tenacity.before_sleep_log(_LOG, logging.WARNING),
)


class YFinanceMarketProvider(IMarketProvider):
    """Yahoo Finance market-data provider built on *yfinance*.

    Fully **stateless** ⇒ thread-safe; create once and share.
    """

    # cache of Yahoo’s reported source time-zones (ticker ➜ tz-name)
    _SOURCE_TZ_CACHE: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @retry_policy
    def fetch_price_data(
        self,
        ticker: str,
        *,
        start: Optional[datetime] = None,
        end:   Optional[datetime] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        tz: str | None = "UTC",
        auto_adjust: bool = False,
        keep_index: bool = True,
    ) -> pd.DataFrame:
        """Return an OHLCV DataFrame indexed by **tz-aware** datetimes.

        Parameters
        ----------
        ticker : str
            Yahoo ticker (case-insensitive).
        start, end : datetime | None
            Boundaries (inclusive, tz-aware). Mutually exclusive with *period*.
        period : str | None
            Yahoo quick period string (e.g. "1y"). Used if *start* is None.
        interval : str
            Bar size; must be in ``_ALLOWED_INTERVALS``.
        tz : str | None
            Time-zone for returned index (``None`` ⇒ keep Yahoo’s tz).
        auto_adjust : bool
            If *True*, return split/dividend adjusted prices.
        keep_index : bool
            If *False*, reset the index to a column ``date_time``.
        """
        # ------------------------ validation -------------------------- #
        if period and (start or end):
            raise ValueError("Specify either (start/end) **or** period, not both.")

        if interval not in _ALLOWED_INTERVALS:
            raise ValueError(f"Interval '{interval}' not supported by Yahoo.")

        if not period and not start:
            period = "1y"  # sensible default

        start = self._ensure_aware(start)
        end   = self._ensure_aware(end)

        self._warn_if_intraday_window_exceeds_limit(interval, start, period)

        # ---------------------------- fetch --------------------------- #
        try:
            yf_ticker = yf.Ticker(ticker)
            raw = yf_ticker.history(
                start=start,
                end=end,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=False,
            )
        except Exception as exc:
            _LOG.error("yfinance error for %s: %s", ticker, exc, exc_info=True,
                       extra={"ticker": ticker, "interval": interval})
            raise

        if raw.empty:
            _LOG.warning(
                "Yahoo returned no data",
                extra={"ticker": ticker, "interval": interval, "period": period},
            )
            return raw

        # -------------------- tz & column normalise -------------------- #
        if tz is not None:
            raw.index = raw.index.tz_convert(ZoneInfo(tz))

        ohlcv = ("Open", "High", "Low", "Close", "Volume")
        missing = set(ohlcv) - set(raw.columns)
        if missing:
            _LOG.warning("Missing columns %s for %s", missing, ticker)

        df = raw[[c for c in ohlcv if c in raw.columns]].copy()
        df.columns = [c.lower() for c in df.columns]
        df.index.name = "date_time"

        if not keep_index:
            df = df.reset_index()

        return df

    # ------------------------------------------------------------------ #
    # Yahoo auxiliary endpoints                                          #
    # ------------------------------------------------------------------ #
    # @retry_policy
    # def fetch_dividends(self, ticker: str, tz: str | None = "UTC") -> pd.Series:
    #     """Return a tz-aware Series of historical dividends."""
    #     try:
    #         series = yf.Ticker(ticker).dividends
    #     except Exception as exc:
    #         _LOG.error("yfinance dividends failed for %s: %s", ticker, exc,
    #                    exc_info=True)
    #         return pd.Series(dtype=float)
    #
    #     if tz is not None and not series.empty:
    #         series.index = series.index.tz_convert(ZoneInfo(tz))
    #     return series

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_aware(dt: Optional[datetime]) -> Optional[datetime]:
        if dt is None:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    def _warn_if_intraday_window_exceeds_limit(
        self,
        interval: str,
        start: Optional[datetime],
        period: Optional[str],
    ) -> None:
        """Emit a warning if the user asks for data Yahoo will not honour."""
        if interval in {"1m"}:
            limit_days = 7
        elif interval in {"2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
            limit_days = 30
        else:
            return  # daily+ has no documented limit

        # compute desired span
        if period:
            try:
                span = pd.Timedelta(period).days
            except ValueError:  # e.g. "1y"
                return
        elif start:
            span = (datetime.now(tz=timezone.utc) - start).days
        else:
            return

        if span > limit_days:
            _LOG.warning(
                "Yahoo limits %s data to ~%sd; request spans %sd",
                interval,
                limit_days,
                span,
            )
