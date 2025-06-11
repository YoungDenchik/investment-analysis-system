# services/forecast_service.py
import logging
import pandas as pd
from typing import Any

from services.price_data.price_data_service import PriceDataService
from services.forecasts.model_manager_final import ModelManager
from datetime import datetime

logger = logging.getLogger(__name__)

class ForecastService:
    """
    Simplified service that fetches data and predicts without meta-learning.
    """

    def __init__(self,
                 price_data: PriceDataService,
                 model_manager: ModelManager
                 ):
        self.price_data = price_data
        self.model_manager = model_manager

    def generate_forecast(self,
                          ticker: str,
                          interval:str,
                          forecast_date: str,
                          **kwargs: Any) -> pd.Series:

        dt = datetime.strptime(forecast_date, "%Y-%m-%d %H:%M:%S")
        df = self.price_data.get_prices(ticker=ticker, interval=interval, end=dt, period='100d')
        predictions = self.model_manager.predict(ticker, df)
        return predictions
