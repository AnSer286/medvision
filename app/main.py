"""MedVision — веб-сервис классификации медицинских изображений.

Курсовая работа, вариант 25.
Дисциплина «Методы и технологии программирования».
"""
from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from app.config import get_settings
from app.gradcam import GradCAM, overlay_heatmap
from app.model import Classifier
from app.preprocessing import preprocess
from app.schemas import HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели один раз при старте сервиса."""
    app.state.classifier = Classifier(
        class_names=settings.class_names,
        weights_path=settings.weights_path,
        device=settings.device,
    )
    yield


app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)


async def _read_validated(file: UploadFile) -> bytes:
    """Прочитать загруженный файл с проверкой типа и размера."""
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"Недопустимый тип файла: {file.content_type}. "
            f"Разрешены: {', '.join(settings.allowed_content_types)}",
        )
    data = await file.read()
    if len(data) > settings.max_upload_size:
        raise HTTPException(status_code=413, detail="Файл превышает 10 МБ")
    return data


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.version,
        model_loaded=hasattr(app.state, "classifier"),
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Классифицировать медицинский снимок (рентген грудной клетки)."""
    data = await _read_validated(file)
    try:
        batch, _ = preprocess(data, settings.image_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    clf: Classifier = app.state.classifier
    probs = clf.predict(batch)
    best = max(probs, key=probs.get)
    return PredictionResponse(
        predicted_class=best,
        confidence=probs[best],
        probabilities=probs,
        fine_tuned=clf.fine_tuned,
    )


@app.post("/api/v1/explain")
async def explain(file: UploadFile = File(...)) -> Response:
    """Вернуть PNG с Grad-CAM-визуализацией областей внимания модели."""
    data = await _read_validated(file)
    try:
        batch, original = preprocess(data, settings.image_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    clf: Classifier = app.state.classifier
    cam_engine = GradCAM(clf.model, clf.target_layer())
    try:
        cam = cam_engine(batch)
    finally:
        cam_engine.remove_hooks()

    png = overlay_heatmap(original, cam)
    return Response(content=png, media_type="image/png")


@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    """Классификация + объяснение одним запросом (для веб-интерфейса)."""
    data = await _read_validated(file)
    try:
        batch, original = preprocess(data, settings.image_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    clf: Classifier = app.state.classifier
    probs = clf.predict(batch)
    best = max(probs, key=probs.get)

    cam_engine = GradCAM(clf.model, clf.target_layer())
    try:
        cam = cam_engine(batch)
    finally:
        cam_engine.remove_hooks()
    png = overlay_heatmap(original, cam)

    return {
        "predicted_class": best,
        "confidence": probs[best],
        "probabilities": probs,
        "fine_tuned": clf.fine_tuned,
        "gradcam_png_base64": base64.b64encode(png).decode(),
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    from pathlib import Path

    html = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))
