import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Final, Literal

import pandas as pd
import pytz
import yfinance as yf

# Initialise logger for this module (inherits root config in application entry‑point)
logger: Final[logging.Logger] = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1) PUBLIC INTERFACE
# ---------------------------------------------------------------------------

class IMarketProvider(ABC):
    """Abstract base‑class that defines the contract for market‑data providers.

    FinTech applications often interact with multiple upstream data vendors.
    A common, stable interface makes it possible to swap providers (e.g. for
    redundancy or cost optimisation) without changing business‑logic code.

    *All* concrete providers **must** honour the semantics documented here.
    """

    # ---------------------------------------------------------------------
    # Primary API
    # ---------------------------------------------------------------------

    @abstractmethod
    def fetch_price_data(
        self,
        ticker: str,
        *,
        # Either a (start, end) range **or** a period string must be supplied
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        period: Optional[str] = None,
        # Granularity; see yfinance docs for valid values ("1m", "15m", "1h", "1d", …)
        interval: str = "1d",
        # Convert the timestamp index to this zone.  ``None`` ⇒ keep source zone.
        tz: Optional[str] = "UTC",
        # Whether to automatically adjust Close prices for corporate actions.
        auto_adjust: bool = False,
        # If ``True`` the "Date" column is left in the index; if ``False`` it is
        # reset to a regular column so that the result is a *flat* DataFrame.
        keep_index: bool = True,
    ) -> pd.DataFrame:
        """Retrieve OHLCV price data.

        Returned *must* contain the columns:
        ``['Open', 'High', 'Low', 'Close', 'Volume']`` (case sensitive).  An
        optional ``'Adj Close'`` column is allowed if *auto_adjust=False*.

        Parameters
        ----------
        ticker : str
            Exchange symbol (e.g. "AAPL").
        start, end : datetime | None
            Inclusive date range (aware or naive).  If *timezone‑naive* they are
            assumed to be in the provider's canonical zone (typically the local
            system zone).  Mutually exclusive with *period*.
        period : str | None
            Text shortcut like "1mo", "5y".  Mutually exclusive with the range
            pair (*start*, *end*).
        interval : str
            Sampling frequency accepted by the underlying API.
        tz : str | None
            IANA time‑zone ("Europe/Kyiv", "UTC", …) to which the index is
            converted.  ``None`` ► no conversion.
        auto_adjust : bool
            Forward‑fill splits & dividends into price series (supported by
            yfinance).
        progress : bool
            Forwarded to yfinance; show download progress bar.
        keep_index : bool
            Whether to keep the datetime index or flatten it into a column
            called "Date".
        """

    # ------------------------------------------------------------------
    # Convenience helpers (optional for implementers, *recommended*)
    # ------------------------------------------------------------------

    def fetch_latest_price(self, ticker: str) -> Optional[float]:
        """Return the most recent *adjusted* close price (or *None* if missing)."""
        df = self.fetch_price_data(ticker, period="1d", interval="1m")
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])

    # Additional helper methods (corporate actions, fundamentals, …) can be
    # added when the project requirements grow.