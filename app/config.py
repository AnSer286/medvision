"""Конфигурация сервиса MedVision."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения (переопределяются переменными окружения MEDVISION_*)."""

    app_name: str = "MedVision — классификация медицинских изображений"
    version: str = "1.0.0"

    # Путь к дообученным весам модели; если файла нет — используется
    # предобученный backbone ImageNet (режим демонстрации).
    weights_path: Path = Path("models/densenet121_chest.pt")

    # Классы модели (бинарная классификация рентгенограмм грудной клетки)
    class_names: tuple[str, ...] = ("NORMAL", "PNEUMONIA")

    # Параметры препроцессинга
    image_size: int = 224

    # Ограничение размера загружаемого файла, байт (10 МБ)
    max_upload_size: int = 10 * 1024 * 1024

    allowed_content_types: tuple[str, ...] = ("image/jpeg", "image/png")

    device: str = "cpu"  # "cuda" при наличии GPU

    model_config = {"env_prefix": "MEDVISION_"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
