from __future__ import annotations

"""
Price‑data service
========================================================
* Facade hiding provider & repository details
* Idempotent bulk‑upsert API (``sync_prices``)
* Gap‑filling read API (``get_prices``)
* Robust timezone handling (UTC‑first)
* Exchange‑aware completeness check via *pandas‑market‑calendars*
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import Final, Optional
import logging
import time

import pandas as pd
import pandas_market_calendars as mcal
from zoneinfo import ZoneInfo

from data.providers.interfaces.i_market_provider import IMarketProvider
from data.persistence.interfaces.repositories.i_price_data_repository import (
    IPriceDataRepository,
)
from data.persistence.interfaces.repositories.i_instrument_repository import (
    IInstrumentRepository,
)

# ---------------------------------------------------------------------------
# Constants & logger
# ---------------------------------------------------------------------------
UTC: Final = timezone.utc
DEFAULT_CAL: Final = "NYSE"
_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry decorator (exponential back‑off)
# ---------------------------------------------------------------------------

def retry(
    attempts: int = 3,
    backoff: float = 1.5,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """Simple retry decorator with exponential back‑off."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):  # type: ignore[override]
            wait = 1.0
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:  # pragma: no cover – generic wrapper
                    if attempt == attempts:
                        _LOG.error("Retry exhausted (%s/%s)", attempt, attempts)
                        raise
                    _LOG.warning(
                        "Transient failure on attempt %s/%s: %s",
                        attempt,
                        attempts,
                        exc,
                    )
                    time.sleep(wait)
                    wait *= backoff

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class PriceDataService:
    """High‑level API for downloading, caching and serving OHLCV data."""

    provider: IMarketProvider
    price_repo: IPriceDataRepository
    instr_repo: IInstrumentRepository
    default_tz: str = "UTC"
    auto_create_instrument: bool = True

    # ---------------------------------------------------------------------
    # WRITE SIDE
    # ---------------------------------------------------------------------

    @retry(attempts=4, backoff=2.0)
    def sync_prices(
        self,
        *,
        ticker: str,
        instrument_id: Optional[int] = None,
        interval: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        period: Optional[str] = "1y",
        auto_adjust: bool = False,
        return_df: bool = False,
    ) -> pd.DataFrame | int:
        """Download prices from *provider* and upsert them to the repository.

        The call is **idempotent** – duplicate rows are resolved with ON CONFLICT.
        """

        instrument_id = instrument_id or self._resolve_instrument_id(ticker)

        df = self.provider.fetch_price_data(
            ticker=ticker,
            start=start,
            end=end,
            period=period,
            interval=interval,
            tz="UTC",
            auto_adjust=auto_adjust,
            keep_index=True,
        )
        if df.empty:
            _LOG.warning("Provider returned 0 rows for %s", ticker)
            return 0

        df = self._normalize_df(df)
        affected = self.price_repo.bulk_upsert_df(df, instrument_id, interval)
        return df if return_df else affected

    # ---------------------------------------------------------------------
    # READ SIDE
    # ---------------------------------------------------------------------

    def get_prices(
        self,
        *,
        ticker: str,
        interval: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        period: Optional[str] = None,
        market_calendar: str = DEFAULT_CAL,
        tz: Optional[str] = None,
        auto_download: bool = True,
        auto_adjust: bool = True,
        date_time_index: bool = True,
    ) -> pd.DataFrame:
        """Return OHLCV data, filling gaps from the provider on‑demand."""

        if period and (start or end):
            raise ValueError("Provide either (start, end) or period – not both.")

        # Period → (start, end)
        if period and not start:
            end = datetime.now(tz=UTC)
            start = end - pd.Timedelta(period)

        tz = tz or self.default_tz
        instrument_id = self._resolve_instrument_id(ticker)
        df = self._read_from_store(instrument_id, interval, start, end)
        # print(df.to_string())
        # On‑demand gap filling
        if auto_download and not self._range_covered(
            df, start, end, market_calendar, interval
        ):
            self._fill_gaps(
                instrument_id=instrument_id,
                ticker=ticker,
                interval=interval,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
            )
            df = self._read_from_store(instrument_id, interval, start, end)
        if df.empty:
            return df

        # print(df)
        # Convert index to user TZ and optionally expose it as a column
        df = df.tz_convert(ZoneInfo(tz))
        # print(df)
        if not date_time_index:
            # df = df.assign(date_time=df.index).set_index("date_time")
            df = df.reset_index()
        # print(df.to_string())
        return df

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize provider output → UTC, ascending index, canonical cols."""

        idx = pd.to_datetime(df.index)
        if idx.tz is None:
            idx = idx.tz_localize(UTC)
        else:
            idx = idx.tz_convert(UTC)

        df = df.copy()
        df.index = idx
        df.sort_index(inplace=True)
        return df

    def _read_from_store(
        self,
        instrument_id: int,
        interval: str,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> pd.DataFrame:
        df = self.price_repo.get_price_data(
            instrument_id=instrument_id,
            interval=interval,
            start_date=start,
            end_date=end,
            as_dataframe=True,
        )

        # # Ensure UTC index *and* keep original for optional "date_time_index"
        # idx = pd.to_datetime(df.pop("date_time"), utc=True)
        # df.index = idx
        # df.rename_axis("ts", inplace=True)
        # df["date_time"] = df.index  # for later use if required
        df.set_index('date_time', inplace=True)

        return df

    @staticmethod
    def _range_covered(
        df: pd.DataFrame,
        start: Optional[datetime],
        end: Optional[datetime],
        market_calendar: str,
        interval: str,
    ) -> bool:
        """True iff *df* covers every expected bar in the [start, end] range."""

        # print(df.to_string())
        # print(df.empty)
        if df.empty or not start or not end:
            return False

        cal = mcal.get_calendar(market_calendar)
        sched = cal.schedule(start_date=start, end_date=end)

        # ------------------------------------------------------------------
        # Daily bars – one per trading session
        # ------------------------------------------------------------------
        if interval.endswith("d"):
            expected = sched.index
            # Localise if naive, else convert
            expected = expected.tz_localize(UTC) if expected.tz is None else expected.tz_convert(UTC)
            present = df.index.normalize().unique()
        # ------------------------------------------------------------------
        # Intraday bars – generate expected timestamps per session
        # ------------------------------------------------------------------
        else:
            expected = pd.DatetimeIndex(
                sorted(
                    ts
                    for open_, close_ in zip(
                        sched["market_open"].dt.tz_convert(UTC),
                        sched["market_close"].dt.tz_convert(UTC),
                    )
                    for ts in pd.date_range(open_, close_, freq=interval, tz=UTC)
                )
            )
            present = df.index.unique()

        missing = expected.difference(present)
        if not missing.empty:
            _LOG.debug("Missing %s bars between %s and %s", len(missing), start, end)
        return missing.empty

    # ------------------------------------------------------------------
    # Internal – provider download wrapper
    # ------------------------------------------------------------------

    def _fill_gaps(
        self,
        *,
        instrument_id: int,
        ticker: str,
        interval: str,
        start: datetime,
        end: datetime,
        auto_adjust: bool = True,
    ) -> None:
        _LOG.info("Filling gaps for %s (%s) from %s to %s", ticker, interval, start, end)
        self.sync_prices(
            ticker=ticker,
            instrument_id=instrument_id,
            interval=interval,
            start=start,
            end=end,
            period=None,
            auto_adjust=auto_adjust,
        )

    # ------------------------------------------------------------------
    # Instrument helper
    # ------------------------------------------------------------------

    def _resolve_instrument_id(self, ticker: str) -> int:
        inst = self.instr_repo.get_by_ticker(ticker)
        if inst:
            return inst.id
        if not self.auto_create_instrument:
            raise ValueError(f"Unknown ticker: {ticker}")
        created = self.instr_repo.create(ticker=ticker)
        _LOG.info("Auto‑created instrument %s → id=%s", ticker, created.id)
        return created.id

    # ------------------------------------------------------------------
    # Administrative helper
    # ------------------------------------------------------------------

    def purge_prices(
        self,
        *,
        instrument_id: int,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> int:
        """Hard‑delete rows from *price_repo* (irreversible)."""

        return self.price_repo.delete_price_data(
            instrument_id=instrument_id,
            interval=interval,
            start_date=start,
            end_date=end,
        )
