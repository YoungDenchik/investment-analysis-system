# api/routers/forecast.py

from fastapi import APIRouter, Depends

# from api.dependencies import get_forecast_service
from services.forecasts.forecast_service import ForecastService
from api.schemas.forecast import ForecastRequest, ForecastResponse
from dependency_injector.wiring import inject, Provide
from di.container import Container

router = APIRouter()

@router.post("/", response_model=ForecastResponse)
def generate_forecast(request: ForecastRequest,
                      service: ForecastService = Depends(
                          Provide[Container.forecast_service]
                      ),
):
    """
    Generates a forecast for the given ticker and date range.
    # """
    predictions = service.generate_forecast(
        ticker=request.ticker,
        forecast_date=request.date,
        interval="1d"
    )
    prediction = predictions.iloc[-1]
    return ForecastResponse(
        ticker=request.ticker,
        date=request.date,
        prediction=prediction
    )
