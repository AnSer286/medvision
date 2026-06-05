"""Pydantic-схемы API."""
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Результат классификации снимка."""

    predicted_class: str = Field(description="Класс с максимальной вероятностью")
    confidence: float = Field(ge=0, le=1, description="Вероятность предсказанного класса")
    probabilities: dict[str, float] = Field(description="Вероятности всех классов")
    fine_tuned: bool = Field(
        description="True — загружены дообученные веса; False — демонстрационный режим"
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
