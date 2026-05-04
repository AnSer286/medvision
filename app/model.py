"""Загрузка и инференс классификационной модели (DenseNet-121)."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from torchvision import models

logger = logging.getLogger(__name__)


def build_model(num_classes: int, pretrained_backbone: bool = True) -> nn.Module:
    """Создать DenseNet-121 с классификатором на num_classes классов."""
    weights = models.DenseNet121_Weights.DEFAULT if pretrained_backbone else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


class Classifier:
    """Обёртка над моделью: загрузка весов, предсказание вероятностей."""

    def __init__(
        self,
        class_names: tuple[str, ...],
        weights_path: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.class_names = class_names
        self.device = torch.device(device)
        self.model = build_model(num_classes=len(class_names))
        self.fine_tuned = False

        if weights_path is not None and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.fine_tuned = True
            logger.info("Загружены дообученные веса: %s", weights_path)
        else:
            logger.warning(
                "Файл весов %s не найден — используется backbone ImageNet "
                "(демонстрационный режим, предсказания не имеют клинической ценности)",
                weights_path,
            )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, batch: torch.Tensor) -> dict[str, float]:
        """Вернуть вероятности классов для батча из одного изображения."""
        logits = self.model(batch.to(self.device))
        probs = torch.softmax(logits, dim=1)[0]
        return {name: float(p) for name, p in zip(self.class_names, probs)}

    def target_layer(self) -> nn.Module:
        """Последний свёрточный блок — целевой слой для Grad-CAM."""
        return self.model.features.denseblock4
