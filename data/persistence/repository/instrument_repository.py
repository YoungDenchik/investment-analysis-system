from __future__ import annotations


from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final, List, Optional, TypeAlias

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from data.persistence.models.instrument import Instrument
from data.persistence.interfaces.repositories.i_instrument_repository import (
    IInstrumentRepository,
)

_LOG: Final = logging.getLogger(__name__)

# Callable that yields a Session (factory or context-manager)
SessionFactory: TypeAlias = Callable[[], Session] | AbstractContextManager[Session]


class InstrumentRepository(IInstrumentRepository):
    """PostgreSQL repository for :class:`Instrument` (SQLAlchemy 2.0 style)."""


    def __init__(self, session_factory: SessionFactory):
        self._session_factory = session_factory

    def _session_ctx(self):
        return self._session_factory()

    # ------------------------------------------------------------------ #
    # CREATE / GET-OR-CREATE                                             #
    # ------------------------------------------------------------------ #
    def create(
        self,
        *,
        ticker: str,
        company_name: str | None = None,
        sector: str | None = None,
        industry: str | None = None,
        currency: str | None = None,
        country: str | None = None,
    ) -> Instrument:
        """Insert and return a new :class:`Instrument` (UPSERT-safe)."""
        # Normalise
        ticker = ticker.upper()

        stmt = (
            pg_insert(Instrument)
            .values(
                ticker=ticker,
                company_name=company_name,
                sector=sector,
                industry=industry,
                currency=currency,
                country=country,
            )
            .on_conflict_do_nothing(index_elements=["ticker"])
            .returning(Instrument)
        )

        # with self._get_session() as s:
        with self._session_ctx() as s:
            try:
                result = s.execute(stmt)
                inst = result.scalar_one_or_none()
                if inst is None:  # already existed
                    inst = s.scalar(select(Instrument).where(Instrument.ticker == ticker))
                s.commit()
                _LOG.info("instrument.create", extra={"ticker": ticker, "id": inst.id})
                return inst
            except SQLAlchemyError:
                s.rollback()
                _LOG.exception("instrument.create failed – rolled back")
                raise

    # ------------------------------------------------------------------ #
    # READ                                                               #
    # ------------------------------------------------------------------ #
    def get_by_id(self, instrument_id: int) -> Optional[Instrument]:
        # with self._get_session() as s:
        with self._session_ctx() as s:
            return s.get(Instrument, instrument_id)

    def get_by_ticker(self, ticker: str) -> Optional[Instrument]:
        # with self._get_session() as s:
        with self._session_ctx() as s:
            return s.scalar(
                select(Instrument).where(Instrument.ticker == ticker.upper())
            )

    def list_all(self, limit: int | None = None) -> List[Instrument]:
        stmt = select(Instrument).order_by(Instrument.id)
        if limit:
            stmt = stmt.limit(limit)
        # with self._get_session() as s:
        with self._session_ctx() as s:
            return list(s.scalars(stmt))

    # ------------------------------------------------------------------ #
    # UPDATE (partial)                                                   #
    # ------------------------------------------------------------------ #
    def update(
        self, instrument_id: int, **fields
    ) -> Instrument:
        """Patch mutable fields; raises if row missing."""
        allowed = {"company_name", "sector", "industry", "currency", "country"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Unknown fields: {bad}")

        # with self._get_session() as s:
        with self._session_ctx() as s:
            inst = s.get(Instrument, instrument_id)
            if not inst:
                raise ValueError(f"Instrument id={instrument_id} not found")

            for k, v in fields.items():
                setattr(inst, k, v)
            s.commit()
            s.refresh(inst)  # update in memory

            _LOG.info("instrument.update", extra={"id": instrument_id, **fields})
            return inst

    # ------------------------------------------------------------------ #
    # DELETE                                                             #
    # ------------------------------------------------------------------ #
    def delete(self, instrument_id: int) -> int:
        # with self._get_session() as s:
        with self._session_ctx() as s:
            try:
                stmt = delete(Instrument).where(Instrument.id == instrument_id)
                result = s.execute(stmt)
                s.commit()
                rows = int(result.rowcount or 0)
                _LOG.info("instrument.delete", extra={"id": instrument_id, "rows": rows})
                return rows
            except SQLAlchemyError:
                s.rollback()
                _LOG.exception("instrument.delete failed – rolled back")
                raise
