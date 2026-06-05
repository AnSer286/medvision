"""Интеграционные тесты API (TestClient, модель загружается реально)."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # контекст запускает lifespan -> загрузку модели
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_probabilities(client, sample_jpeg_bytes):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("xray.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["probabilities"]) == {"NORMAL", "PNEUMONIA"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-5
    assert body["predicted_class"] in body["probabilities"]


def test_predict_rejects_wrong_type(client):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("doc.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415


def test_predict_rejects_corrupt_image(client):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("bad.png", b"not a real png", "image/png")},
    )
    assert resp.status_code == 422


def test_explain_returns_png(client, sample_jpeg_bytes):
    resp = client.post(
        "/api/v1/explain",
        files={"file": ("xray.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_analyze_combined(client, sample_jpeg_bytes):
    resp = client.post(
        "/api/v1/analyze",
        files={"file": ("xray.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "gradcam_png_base64" in body
    assert body["predicted_class"] in {"NORMAL", "PNEUMONIA"}
