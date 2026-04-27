"""Препроцессинг входных изображений."""
from __future__ import annotations

import io

import torch
from PIL import Image
from torchvision import transforms

# Нормализация ImageNet — стандарт для предобученных моделей torchvision
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int = 224) -> transforms.Compose:
    """Трансформации инференса: resize -> center crop -> tensor -> normalize."""
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_image(data: bytes) -> Image.Image:
    """Открыть изображение из байтов и привести к RGB.

    Raises:
        ValueError: если данные не являются корректным изображением.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001 — PIL бросает разные типы
        raise ValueError("Файл не является корректным изображением") from exc
    return image.convert("RGB")


def preprocess(data: bytes, image_size: int = 224) -> tuple[torch.Tensor, Image.Image]:
    """Байты изображения -> (батч-тензор [1,3,H,W], исходное PIL-изображение)."""
    image = load_image(data)
    tensor = build_transform(image_size)(image).unsqueeze(0)
    return tensor, image
