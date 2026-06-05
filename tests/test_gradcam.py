"""Тесты Grad-CAM на компактной CNN (быстро, без DenseNet)."""
import numpy as np
import torch
from PIL import Image
from torch import nn

from app.gradcam import GradCAM, overlay_heatmap


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.head(self.conv(x))


def test_gradcam_output_range_and_shape():
    model = TinyCNN()
    cam_engine = GradCAM(model, model.conv[3])  # последний conv-слой
    batch = torch.rand(1, 3, 32, 32)
    cam = cam_engine(batch)
    cam_engine.remove_hooks()

    assert cam.shape == (32, 32)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0


def test_gradcam_explicit_class():
    model = TinyCNN()
    cam_engine = GradCAM(model, model.conv[3])
    cam = cam_engine(torch.rand(1, 3, 32, 32), class_idx=1)
    cam_engine.remove_hooks()
    assert cam.shape == (32, 32)


def test_overlay_heatmap_returns_png():
    image = Image.new("RGB", (64, 64), color=(100, 100, 100))
    cam = np.random.rand(32, 32).astype(np.float32)
    png = overlay_heatmap(image, cam)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # сигнатура PNG
