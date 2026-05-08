"""Grad-CAM: визуальная объяснимость решений CNN.

Реализация по статье Selvaraju et al., "Grad-CAM: Visual Explanations
from Deep Networks via Gradient-based Localization" (ICCV 2017).
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn


class GradCAM:
    """Вычисляет карту активации классов через градиенты целевого слоя."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, args, output) -> None:  # noqa: ANN001
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:  # noqa: ANN001
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()

    def __call__(self, batch: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Карта значимости [H, W] в диапазоне [0, 1] для одного изображения.

        Args:
            batch: тензор [1, 3, H, W].
            class_idx: индекс целевого класса; по умолчанию — argmax.
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(batch)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Hooks не сработали: проверьте целевой слой")

        # Глобальное усреднение градиентов -> веса каналов (alpha_k)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.relu((weights * self._activations).sum(dim=1))[0]  # [h, w]

        cam_np = cam.cpu().numpy()
        cam_np -= cam_np.min()
        if cam_np.max() > 0:
            cam_np /= cam_np.max()

        h, w = batch.shape[2], batch.shape[3]
        return cv2.resize(cam_np, (w, h))


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> bytes:
    """Наложить тепловую карту на исходное изображение, вернуть PNG-байты."""
    img = np.array(image.convert("RGB"))
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return buf.getvalue()
