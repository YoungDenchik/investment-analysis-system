#data/persistence/repository/i_price_data_repository.py
from __future__ import annotations

import logging
from collections.abc import Iterable, Callable
from datetime import datetime, timezone
from typing import Final, List, TypeAlias

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from data.persistence.models.price_data import PriceData
from data.persistence.interfaces.repositories.i_price_data_repository import (
    IPriceDataRepository,
)

_LOG: Final = logging.getLogger(__name__)
UTC = timezone.utc

# --------------------------------------------------------------------------- #
# Helper typing – lets us annotate any factory that returns a Session         #
# --------------------------------------------------------------------------- #
SessionFactory: TypeAlias = Callable[[], Session]


class PriceDataRepository(IPriceDataRepository):
    """PostgreSQL repository for :class:`PriceData` (SQLAlchemy 2.0 style)."""

    def __init__(self, session_factory: SessionFactory):
        self._session_factory = session_factory

    def _session_ctx(self):
        return self._session_factory()

    # ----------------------------- Public API -------------------------- #
    def bulk_upsert(self, records: Iterable[PriceData]) -> int:
        """Insert **or** update (idempotent) and return affected rows."""
        records = list(records)
        if not records:
            _LOG.debug("bulk_upsert: empty input – skip")
            return 0

        stmt = pg_insert(PriceData).values(
            [
                {
                    "instrument_id": r.instrument_id,
                    "date_time": r.date_time,
                    "interval": r.interval,
                    "open": float(r.open),
                    "high": float(r.high),
                    "low": float(r.low),
                    "close": float(r.close),
                    "volume": int(r.volume) if r.volume is not None else None,
                }
                for r in records
            ]
        )
        update_cols = {
            c.name: c
            for c in stmt.excluded
            if c.name not in ("instrument_id", "date_time", "interval")
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["instrument_id", "date_time", "interval"],
            set_=update_cols,
        )  # SQLAlchemy’s canonical PostgreSQL UPSERT :contentReference[oaicite:5]{index=5}

        # with self._get_session() as s:
        with self._session_ctx() as s:
            try:
                result = s.execute(stmt)
                s.commit()
                _LOG.info(
                    "bulk_upsert",
                    extra={"rows": result.rowcount, "table": PriceData.__tablename__},
                )
                return int(result.rowcount or 0)
            except SQLAlchemyError:
                s.rollback()
                _LOG.exception("bulk_upsert failed – rolled back")
                raise

    def bulk_upsert_df(
        self, df: pd.DataFrame, instrument_id: int, interval: str
    ) -> int:
        """Convenience wrapper for Pandas users."""
        records = PriceData.from_dataframe(df, instrument_id, interval)
        return self.bulk_upsert(records)

    def get_price_data(
            self,
            instrument_id: int,
            interval: str,
            start_date: datetime,
            end_date: datetime,
            *,
            as_dataframe: bool = True,
    ) -> pd.DataFrame | List[PriceData]:
        """
        Повертає записи (start_date, end_date] для інструмента й інтервалу.
        Якщо as_dataframe=True, віддає DataFrame з колонками:
          date_time (tz-aware UTC), open, high, low, close, volume
        """
        # 1) Формуємо базовий фільтр
        base_filter = (
                (PriceData.instrument_id == instrument_id)
                & (PriceData.interval == interval)
                & PriceData.date_time.between(start_date, end_date)
        )

        # 2) Якщо потрібен DataFrame
        if as_dataframe:
            # вибираємо лише потрібні стовпці
            qry = (
                select(
                    PriceData.date_time,
                    PriceData.open,
                    PriceData.high,
                    PriceData.low,
                    PriceData.close,
                    PriceData.volume,
                )
                .where(base_filter)
                .order_by(PriceData.date_time)
            )

            # with self._get_session() as s:
            with self._session_ctx() as s:
                df = pd.read_sql(
                    qry,
                    con=s.bind,
                    # parse_dates=["date_time"],  # перетворює в datetime64
                )

            df['date_time'] = pd.to_datetime(df['date_time'], utc=True)
            # df['date_time'] = df['date_time'].dt.tz_localize('UTC')

            # print(2)
            # print(df.dtypes)
            # print(df.to_string())

            # Якщо часи прийшли без tz, локалізуємо в UTC
            if df["date_time"].dt.tz is None:
                df["date_time"] = df["date_time"].dt.tz_localize("UTC")

            def _to_int(v):
                if isinstance(v, (bytes, bytearray)):
                    # найчастіше DB віддає little-endian
                    return int.from_bytes(v, byteorder="little", signed=True)
                return int(v) if pd.notna(v) else None

            df["volume"] = df["volume"].apply(_to_int)

            return df

        # 3) Інакше – віддаємо список ORM-екземплярів,
        #    але завантажуємо лише потрібні атрибути
        from sqlalchemy.orm import load_only

        qry = (
            select(PriceData)
            .options(load_only(
                "date_time", "open", "high", "low", "close", "volume"
            ))
            .where(base_filter)
            .order_by(PriceData.date_time)
        )
        # with self._get_session() as s:
        with self._session_ctx() as s:
            rows = list(s.scalars(qry))
        return rows

    def delete_price_data(
        self,
        instrument_id: int,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> int:
        """Hard-delete rows and return affected count (no soft-deletes)."""
        # with self._get_session() as s:
        with self._session_ctx() as s:
            try:
                stmt = (
                    delete(PriceData)
                    .where(PriceData.instrument_id == instrument_id)
                    .where(PriceData.interval == interval)
                    .where(PriceData.date_time.between(start_date, end_date))
                )
                result = s.execute(stmt)
                s.commit()
                rows = int(result.rowcount or 0)
                _LOG.info(
                    "delete_price_data",
                    extra={
                        "rows": rows,
                        "instrument": instrument_id,
                        "partition_hint": "consider partitioning for large windows",
                    },
                )  # partitioning beats large deletes :contentReference[oaicite:7]{index=7}
                return rows
            except SQLAlchemyError:
                s.rollback()
                _LOG.exception("delete_price_data failed – rolled back")
                raise