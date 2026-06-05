"""Общие фикстуры тестов."""
import io

import pytest
from PIL import Image


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Синтетическое изображение 64x64 в формате PNG."""
    img = Image.new("RGB", (64, 64), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    img = Image.new("L", (96, 96), color=200)  # grayscale, как рентген
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
