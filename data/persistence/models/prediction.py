# prediction.py
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from data.persistence.models.base import Base


class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    instrument_id = Column(Integer, ForeignKey('instruments.id'))
    date = Column(DateTime, nullable=False)  # Для якої дати зроблено прогноз
    predicted_value = Column(Float, nullable=False)
    model_name = Column(String, nullable=False)  # Наприклад, назва/версія моделі
    confidence = Column(Float, nullable=True)  # Якщо потрібен якийсь індекс впевненості

    # Можна додати relationship, якщо потрібно:
    # instrument = relationship("Instrument", backref="predictions")

    def __repr__(self):
        return f"<Prediction(id={self.id}, instrument_id={self.instrument_id}, date={self.date})>"
