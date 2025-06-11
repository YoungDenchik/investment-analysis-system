# data/persistence/interfaces/repositories/i_price_data_repository.py

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence, Union

import pandas as pd

from data.persistence.models.price_data import PriceData

class IPriceDataRepository(ABC):
    """
    Інтерфейс для репозиторія PriceData.
    """

    @abstractmethod
    def bulk_upsert(self, records: Sequence[PriceData]) -> int:
        """
        Масово вставляє або оновлює записи PriceData.
        :param records: Список моделей PriceData.
        :return: Кількість оброблених рядків.
        """
        ...

    @abstractmethod
    def bulk_upsert_df(self, df: pd.DataFrame, instrument_id: int, interval: str) -> int:
        """
        Обгортка для bulk_upsert, що приймає DataFrame.
        :param df: Дані у форматі pd.DataFrame.
        :param instrument_id: Ідентифікатор інструмента.
        :param interval: Інтервал (наприклад, '1d', '1h').
        :return: Кількість оброблених рядків.
        """
        ...

    @abstractmethod
    def get_price_data(
        self,
        instrument_id: int,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        as_dataframe: bool = True
    ) -> Union[pd.DataFrame, Sequence[PriceData]]:
        """
        Повертає дані цін у вказаному діапазоні.
        :param instrument_id: ID інструмента.
        :param interval: Інтервал ('1d', '1h' тощо).
        :param start_date: Початок діапазону (UTC).
        :param end_date: Кінець діапазону (UTC).
        :param as_dataframe: Якщо True — DataFrame, інакше — список моделей.
        """
        ...

    @abstractmethod
    def delete_price_data(
        self,
        instrument_id: int,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """
        Видаляє записи цін у вказаному діапазоні.
        :param instrument_id: ID інструмента.
        :param interval: Інтервал ('1d', '1h' тощо).
        :param start_date: Початок діапазону (UTC).
        :param end_date: Кінець діапазону (UTC).
        :return: Кількість видалених рядків.
        """
        ...
