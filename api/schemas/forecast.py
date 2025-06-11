# api/schemas/forecast.py

from pydantic import BaseModel
from typing import List

class ForecastRequest(BaseModel):
    ticker: str
    date: str

class ForecastResponse(BaseModel):
    ticker: str
    date: str
    prediction: float
