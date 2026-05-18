"""Дообучение DenseNet-121 на датасете Chest X-Ray Pneumonia.

Датасет: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Ожидаемая структура каталога:
    data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg

Запуск:
    python -m train.train --data data/chest_xray --epochs 5 --out models/densenet121_chest.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.model import build_model
from app.preprocessing import IMAGENET_MEAN, IMAGENET_STD


def build_loaders(root: Path, image_size: int, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "test", transform=val_tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=batch_size, num_workers=2),
    )


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("models/densenet121_chest.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(args.data, args.image_size, args.batch_size)

    model = build_model(num_classes=2).to(device)
    # Дообучаем только классификатор и последний блок — быстрее и стабильнее
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.features.denseblock4.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += float(loss)

        acc = evaluate(model, val_loader, device)
        print(f"Эпоха {epoch}: loss={running / len(train_loader):.4f}, test acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  Сохранены лучшие веса -> {args.out}")

    print(f"Готово. Лучшая точность на тесте: {best_acc:.4f}")


if __name__ == "__main__":
    main()
