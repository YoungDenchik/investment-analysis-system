# api/main.py

"""
main.py

Entry point for FastAPI application.
"""

# api/main.py
import uvicorn
from fastapi import FastAPI
from api.routers import forecast
from di.container import Container
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    """
    Створює та конфігурує FastAPI-додаток, включаючи DI-контейнер і маршрути.
    """
    # Ініціалізуємо контейнер залежностей
    container = Container()



    # Створюємо FastAPI-додаток і прикріплюємо контейнер для доступу з роутерів
    app = FastAPI(
        title="Intelligent Investment Analysis System API"
    )
    app.container = container

    # Option A: Allow all origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Підписуємося на події запуску/завершення для ресурсів контейнера
    @app.on_event("startup")
    def startup_event():
        container.init_resources()
        # Wiring: вказуємо пакети з роутерами для DI
        container.wire(modules=[
            "api.main",
            # "api.routers.instruments",
            "api.routers.forecast",
            # "api.routers.recommendations",
        ])

    @app.on_event("shutdown")
    def shutdown_event():
        # Якщо є ресурси з методом shutdown, викликаємо їх
        try:
            container.shutdown_resources()
        except AttributeError:
            pass

    app.include_router(
        forecast.router,
        prefix="/forecast",
        tags=["Forecast"],
    )

    return app


# Створюємо екземпляр додатка для uvicorn
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )