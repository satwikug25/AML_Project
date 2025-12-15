import argparse
import os
import time
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import timm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime



def get_dataloaders(
    data_root: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, Dict[int, str]]:

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    print(f"[DATA] Using data_root = {data_root}")
    print(f"[DATA] Train dir: {train_dir}")
    print(f"[DATA] Val   dir: {val_dir}")
    print(f"[DATA] Test  dir: {test_dir}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    num_classes = len(train_dataset.classes)
    idx_to_class = {i: c for i, c in enumerate(train_dataset.classes)}

    print(f"[DATA] Classes ({num_classes}): {train_dataset.classes}")
    print(f"[DATA] #train images: {len(train_dataset)}")
    print(f"[DATA] #val   images: {len(val_dataset)}")
    print(f"[DATA] #test  images: {len(test_dataset)}")

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    print(f"[DATA] Train batches per epoch: {len(train_loader)}")
    print(f"[DATA] Val   batches per epoch: {len(val_loader)}")
    print(f"[DATA] Test  batches per epoch: {len(test_loader)}")

    return train_loader, val_loader, test_loader, num_classes, idx_to_class

def build_resnet(num_classes: int, variant: str = "resnet18", pretrained: bool = True) -> nn.Module:
    print(f"[MODEL] Building ResNet variant={variant}, pretrained={pretrained}")
    if variant == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.fc.in_features
    elif variant == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.fc.in_features
    else:
        raise ValueError(f"Unsupported ResNet variant: {variant}")

    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_vit(num_classes: int, variant: str = "vit_b16", pretrained: bool = True) -> nn.Module:
    if variant == "vit_b16":
        name = "vit_base_patch16_224"
    elif variant == "vit_s16":
        name = "vit_small_patch16_224"
    else:
        raise ValueError(f"Unsupported ViT variant: {variant}")

    print(f"[MODEL] Building ViT variant={variant} ({name}), pretrained={pretrained}")
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model




def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_interval: int = 10,
) -> float:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(loader)
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            curr_loss = running_loss / total
            curr_acc = correct / total
            print(
                f"[TRAIN] Epoch {epoch}/{total_epochs} "
                f"Batch {batch_idx}/{num_batches} "
                f"Loss: {curr_loss:.4f} Acc: {curr_acc:.4f}"
            )

    epoch_time = time.time() - start_time
    avg_loss = running_loss / total
    acc = correct / total
    print(f"[TRAIN] Epoch {epoch}/{total_epochs} done in {epoch_time:.1f}s "
          f"(avg_loss={avg_loss:.4f}, acc={acc:.4f})")
    return avg_loss, acc


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "VAL",
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_time = time.time() - start_time
    avg_loss = running_loss / total
    acc = correct / total
    print(f"[{split_name}] Eval done in {epoch_time:.1f}s "
          f"(avg_loss={avg_loss:.4f}, acc={acc:.4f})")
    return avg_loss, acc


def test_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
   
    model.eval()
    correct = 0
    total = 0
    all_true = []
    all_pred = []

    num_batches = len(loader)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    epoch_time = time.time() - start_time
    acc = correct / total
    print(f"[TEST] Inference done in {epoch_time:.1f}s (acc={acc:.4f})")
    return acc, np.array(all_true), np.array(all_pred)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_training_plots(result_dir: str, train_losses, val_losses, train_accs, val_accs):
    _ensure_dir(result_dir)
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="train_loss", marker='o')
    plt.plot(epochs, val_losses, label="val_loss", marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="train_acc", marker='o')
    plt.plot(epochs, val_accs, label="val_acc", marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    out_path = os.path.join(result_dir, 'train_val_curves.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_confusion_matrix(result_dir: str, y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: Dict[int, str]):
    _ensure_dir(result_dir)
    cm = confusion_matrix(y_true, y_pred)
    labels = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.title('Confusion Matrix')
    out_path = os.path.join(result_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_classification_report(result_dir: str, y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: Dict[int, str]):
    _ensure_dir(result_dir)
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    report = classification_report(y_true, y_pred, target_names=target_names)
    out_txt = os.path.join(result_dir, 'classification_report.txt')
    with open(out_txt, 'w') as f:
        f.write(report)


def save_metadata(result_dir: str, meta: dict):
    _ensure_dir(result_dir)
    out_json = os.path.join(result_dir, 'metadata.json')
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="medicine_boxes_split",
                        help="Root folder with train/val/test subdirectories")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "vit_b16", "vit_s16"],
                        help="Which model to train")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="If set, do not use ImageNet-pretrained weights")
    args = parser.parse_args()

    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[SETUP] Using device: {device}")
    print(f"[SETUP] Args: {args}")

    train_loader, val_loader, test_loader, num_classes, idx_to_class = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"[SETUP] Found {num_classes} classes:")
    for idx, name in idx_to_class.items():
        print(f"  {idx}: {name}")

    pretrained = not args.no_pretrained
    print(f"[SETUP] Pretrained weights: {pretrained}")

    # Build model
    if args.model.startswith("resnet"):
        model = build_resnet(num_classes, variant=args.model, pretrained=pretrained)
    else:  # vit_b16 or vit_s16
        model = build_vit(num_classes, variant=args.model, pretrained=pretrained)

    model = model.to(device)
    print("[SETUP] Model built and moved to device.")

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    best_state = None
    # Histories for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Current LR (may change over time)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n========== Epoch {epoch}/{args.epochs} (lr={current_lr:.6f}) ==========")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch=epoch, total_epochs=args.epochs, log_interval=10
        )
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, device, split_name="VAL"
        )

        # record histories
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_acc)

        print(
            f"[EPOCH SUMMARY] {epoch}/{args.epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            print(f"[CHECKPOINT] New best val acc: {best_val_acc:.4f} (epoch {epoch})")

    print(f"\n[TRAINING DONE] Best val acc: {best_val_acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
        print("[MODEL] Loaded best validation checkpoint.")

    # Final test evaluation
    test_acc, y_true, y_pred = test_model(model, test_loader, device)
    print(f"\n[RESULT] Test accuracy: {test_acc:.4f}")

    # Confusion matrix & classification report
    cm = confusion_matrix(y_true, y_pred)
    print("\n[RESULT] Confusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\n[RESULT] Classification report:")
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    print(classification_report(y_true, y_pred, target_names=target_names))

    # --- Save results to disk ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", args.model, timestamp)
    _ensure_dir(result_dir)

    # Save best model (state dict)
    ckpt_path = os.path.join(result_dir, 'best_model.pth')
    state_to_save = best_state if best_state is not None else model.state_dict()
    torch.save(state_to_save, ckpt_path)
    print(f"[SAVE] Saved model checkpoint to {ckpt_path}")

    # Save training curves
    save_training_plots(result_dir, train_losses, val_losses, train_accs, val_accs)
    print(f"[SAVE] Saved training curves to {result_dir}")

    # Save confusion matrix and classification report
    save_confusion_matrix(result_dir, y_true, y_pred, idx_to_class)
    save_classification_report(result_dir, y_true, y_pred, idx_to_class)
    print(f"[SAVE] Saved confusion matrix and classification report to {result_dir}")

    # Save metadata
    meta = {
        'args': vars(args),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'timestamp': timestamp,
        'num_classes': num_classes,
    }
    save_metadata(result_dir, meta)
    # Also save idx_to_class mapping
    with open(os.path.join(result_dir, 'idx_to_class.json'), 'w') as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"[SAVE] Saved metadata and mappings to {result_dir}")


if __name__ == "__main__":
    main()
