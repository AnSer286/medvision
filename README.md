# MedVision — сервис классификации медицинских изображений

Курсовая работа (проект), **вариант 25**, дисциплина «Методы и технологии программирования».

Веб-сервис для загрузки медицинских снимков (рентген грудной клетки) и классификации
патологий свёрточной нейросетью **DenseNet-121** с объяснением решений методом **Grad-CAM**.

**Стек:** Python 3.12 · PyTorch · FastAPI · Docker

## Возможности

- Классификация рентгенограмм: `NORMAL` / `PNEUMONIA`
- Объяснимость: тепловая карта Grad-CAM показывает области, на которые «смотрела» модель
- REST API с автодокументацией (Swagger UI на `/docs`)
- Веб-интерфейс с drag-and-drop загрузкой
- Валидация входных файлов (тип, размер, корректность изображения)

## Быстрый старт

### Docker (рекомендуется)

```bash
docker compose up --build
# Веб-интерфейс: http://localhost:8000
# Swagger UI:    http://localhost:8000/docs
```

### Локально

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Без файла весов `models/densenet121_chest.pt` сервис работает в демонстрационном
режиме (backbone ImageNet, предсказания не имеют клинической ценности).

## Обучение модели

Датасет: [Chest X-Ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

```bash
pip install -r requirements-dev.txt
python -m train.train --data data/chest_xray --epochs 5 --out models/densenet121_chest.pt
```

## API

| Метод | Путь              | Описание                                  |
|-------|-------------------|-------------------------------------------|
| GET   | `/health`         | Статус сервиса                            |
| POST  | `/api/v1/predict` | Классификация снимка (JSON)               |
| POST  | `/api/v1/explain` | Grad-CAM-визуализация (PNG)               |
| POST  | `/api/v1/analyze` | Классификация + Grad-CAM одним запросом   |

Пример:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@xray.jpg;type=image/jpeg"
```

```json
{
  "predicted_class": "PNEUMONIA",
  "confidence": 0.93,
  "probabilities": {"NORMAL": 0.07, "PNEUMONIA": 0.93},
  "fine_tuned": true
}
```

## Тестирование и качество кода

```bash
pytest                      # модульные + интеграционные тесты, покрытие
ruff check app tests        # линтер
bandit -r app               # SAST-анализ
pip-audit -r requirements.txt  # проверка уязвимостей зависимостей
```

CI (GitHub Actions): линтер → SAST → аудит зависимостей → тесты → сборка Docker.

## Структура проекта

```
medvision/
├── app/
│   ├── main.py           # FastAPI-приложение, эндпоинты
│   ├── model.py          # DenseNet-121, загрузка весов, инференс
│   ├── gradcam.py        # Grad-CAM и наложение тепловой карты
│   ├── preprocessing.py  # подготовка изображений
│   ├── schemas.py        # Pydantic-схемы
│   ├── config.py         # настройки
│   └── static/index.html # веб-интерфейс
├── train/train.py        # скрипт дообучения
├── tests/                # pytest-тесты
├── docs/                 # архитектура, план работ
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Дисклеймер

Сервис носит **учебный характер** и не является медицинским изделием.
Результаты классификации не могут использоваться для постановки диагноза.
