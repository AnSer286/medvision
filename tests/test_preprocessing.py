"""Тесты препроцессинга изображений."""
import pytest

from app.preprocessing import load_image, preprocess


def test_load_image_valid(sample_image_bytes):
    img = load_image(sample_image_bytes)
    assert img.mode == "RGB"
    assert img.size == (64, 64)


def test_load_image_grayscale_converted_to_rgb(sample_jpeg_bytes):
    img = load_image(sample_jpeg_bytes)
    assert img.mode == "RGB"


def test_load_image_invalid_raises():
    with pytest.raises(ValueError, match="не является корректным изображением"):
        load_image(b"this is not an image")


def test_preprocess_shape(sample_image_bytes):
    tensor, original = preprocess(sample_image_bytes, image_size=224)
    assert tensor.shape == (1, 3, 224, 224)
    assert original.size == (64, 64)


def test_preprocess_normalized(sample_image_bytes):
    tensor, _ = preprocess(sample_image_bytes, image_size=224)
    # После нормализации значения не должны лежать в [0, 1]
    assert float(tensor.min()) < 0
