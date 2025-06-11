# data/persistence/ml_models/price_data.py
from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    UniqueConstraint,
)

from sqlalchemy.orm import Mapped, mapped_column, relationship

from data.persistence.models.base import Base


class PriceData(Base):
    __tablename__ = 'price_data'

    # Припустимо, що є Composite Primary Key: (instrument_id, date, interval)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), primary_key=True
    )

    date_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),  # must be UTC
        primary_key=True,
        comment="Timestamp in UTC (aware)"
    )

    interval: Mapped[str] = mapped_column(String(length=4), primary_key=True)

    open: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=True)


    # Зав’язка на Instrument
    instrument = relationship("Instrument", backref="price_data")


    __table_args__ = (
        UniqueConstraint("instrument_id", "date_time", "interval", name="uix_price_pk"),
        Index("ix_price_instrument_ts", "instrument_id", "date_time"),
    )

    # ------------------------------------------------------------------
    # Converters (DF ↔ ORM) – thin static helpers keep repository clean.
    # ------------------------------------------------------------------

    @staticmethod
    def from_dataframe(df: pd.DataFrame, instrument_id: int, interval: str) -> list["PriceData"]:
        """Convert DataFrame to list[PriceData]. Expects tz‑aware index (UTC)."""
        expected = {"open", "high", "low", "close", "volume"}
        if not expected.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns {expected}")
        records: list[PriceData] = []
        for ts, row in df.iterrows():
            if ts.tzinfo is None:
                raise ValueError("Index must be timezone‑aware (UTC)")
            records.append(
                PriceData(
                    instrument_id=instrument_id,
                    date_time=ts,  # already UTC per service contract
                    interval=interval,
                    open=row.open,
                    high=row.high,
                    low=row.low,
                    close=row.close,
                    volume=row.volume if not pd.isna(row.volume) else None,
                )
            )
        return records

    def __repr__(self) -> str:  # pragma: no cover
        return (
            "<PriceData(inst_id=%s, date_time=%s, interval=%s, close=%s)>"
            % (self.instrument_id, self.date_time.isoformat(), self.interval, self.close)
        )
