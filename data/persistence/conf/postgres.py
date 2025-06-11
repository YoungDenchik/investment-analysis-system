# data/persistence/postgres.py
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
import config.config as cfg


# Engine – один на процес
def get_postgres_engine() -> Engine:
    """
    Ініціалізує (lazy) SQLAlchemy Engine для PostgreSQL, використовуючи 
    налаштування з env змінних.
    """
    user = cfg.POSTGRES_USER
    password = cfg.POSTGRES_PASSWORD
    host = cfg.POSTGRES_HOST
    port = cfg.POSTGRES_PORT
    db_name = cfg.POSTGRES_DB

    url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(url, pool_pre_ping=True, echo=False, future=True)
    return engine


engine = get_postgres_engine()

# Фабрика сесій (expire_on_commit=False щоб DataFrame працювали поза транзакцією)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
