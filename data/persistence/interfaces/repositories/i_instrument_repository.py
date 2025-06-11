# i_instrument_repository.py
# ---------------------------------------------------------------------------
# Interface + PostgreSQL implementation for the Instruments reference table.
# Uses mapper functions to translate between ORM entity & lightweight DTO.
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Iterable, List, Protocol, runtime_checkable, Optional
from data.persistence.models.instrument import Instrument



# ---------------------------------------------------------------------------
# INTERFACE – structural typing via Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IInstrumentRepository(Protocol):
    """Protocol for a repository managing Instrument entities."""

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
        """
        Insert (upsert-safe) and return an Instrument.
        """
        ...

    def get_by_id(self, instrument_id: int) -> Optional[Instrument]:
        """
        Return an Instrument by its ID, or None if not found.
        """
        ...

    def get_by_ticker(self, ticker: str) -> Optional[Instrument]:
        """
        Return an Instrument by its ticker (case-insensitive), or None if not found.
        """
        ...

    def list_all(self, limit: int | None = None) -> List[Instrument]:
        """
        Return all Instruments ordered by ID. If `limit` is given, return at most `limit` items.
        """
        ...

    def update(self, instrument_id: int, **fields) -> Instrument:
        """
        Patch mutable fields on an existing Instrument.
        Raises ValueError if the instrument is missing or unknown fields are provided.
        """
        ...

    def delete(self, instrument_id: int) -> int:
        """
        Delete the Instrument by ID.
        Returns the number of rows deleted (0 or 1).
        """
        ...