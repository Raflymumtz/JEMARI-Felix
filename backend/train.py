"""GPU-accelerated training script for the BISINDO CNN+Transformer model.

Implements BAB 3.3 / 3.4 of the proposal:
  - loads the pre-cropped dataset (see preprocess.py)
  - splits it 70% train / 15% validation / 15% test (chronological, per
    class, so no near-duplicate frame leaks across splits)
  - trains MobileNetV2 (CNN) + Transformer (self-attention) end to end
  - reports accuracy, precision, recall, F1-score and inference latency
    on the held-out test set, exactly the metrics named in the proposal.

Run:
    python train.py
Requires an NVIDIA GPU (VGA) with CUDA; falls back to CPU automatically
if none is found, but that will be far slower.
"""
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config
from dataset import build_splits, SequenceDataset, compute_class_weights
from model import SignCNNTransformer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[GPU] Training on {name} (CUDA {torch.version.cuda})")
        return torch.device("cuda")
    print("[CPU] No CUDA GPU found — training on CPU (this will be slow).")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, scaler, train=True):
    model.train(mode=train)
    total_loss, all_preds, all_labels = 0.0, [], []

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(frames)
                loss = criterion(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * frames.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate_test_set(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        logits = model(frames)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return acc, precision, recall, f1, all_labels, all_preds


@torch.no_grad()
def measure_latency(model, device, n_warmup=10, n_measure=50):
    model.eval()
    dummy = torch.randn(1, config.WINDOW, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)

    for _ in range(n_warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(timings)), float(np.percentile(timings, 95))


def main():
    set_seed(config.SEED)
    device = get_device()
    os.makedirs(config.SAVED_MODEL_DIR, exist_ok=True)

    print("Building train/val/test window splits from processed dataset...")
    train_samples, val_samples, test_samples, label_map, class_counts = build_splits()
    num_classes = len(label_map)
    print(f"Classes ({num_classes}): {list(label_map.values())}")
    print(f"Sequence windows -> train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    train_ds = SequenceDataset(train_samples, augment=True)
    val_ds = SequenceDataset(val_samples, augment=False)
    test_ds = SequenceDataset(test_samples, augment=False)

    num_workers = 4 if os.name != "nt" else 0  # safest default on Windows
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                               num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = SignCNNTransformer(num_classes=num_classes, pretrained=True).to(device)

    class_weights = compute_class_weights(train_samples, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    best_val_f1 = -1.0
    patience, patience_counter = 6, 0
    history = []

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, train=True
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler, train=False
        )
        scheduler.step()
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{config.EPOCHS} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"({dt:.1f}s)"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1, "seconds": dt,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "num_classes": num_classes}, config.MODEL_PATH)
            print(f"  -> new best val_f1={val_f1:.4f}, checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping (no val_f1 improvement in {patience} epochs).")
                break

    with open(config.HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print("\nLoading best checkpoint for final test-set evaluation...")
    checkpoint = torch.load(config.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    acc, precision, recall, f1, y_true, y_pred = evaluate_test_set(model, test_loader, device, num_classes)
    latency_mean, latency_p95 = measure_latency(model, device)

    metrics = {
        "test_accuracy": acc,
        "test_precision_macro": precision,
        "test_recall_macro": recall,
        "test_f1_macro": f1,
        "latency_ms_mean": latency_mean,
        "latency_ms_p95": latency_p95,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "num_classes": num_classes,
        "window": config.WINDOW,
        "img_size": config.IMG_SIZE,
        "train_windows": len(train_samples),
        "val_windows": len(val_samples),
        "test_windows": len(test_samples),
        "raw_frame_counts": class_counts,
    }

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(config.LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print("\n=== TEST SET RESULTS ===")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}% (macro)")
    print(f"Recall:    {recall*100:.2f}% (macro)")
    print(f"F1-score:  {f1*100:.2f}% (macro)")
    print(f"Latency:   {latency_mean:.1f} ms/window avg, {latency_p95:.1f} ms p95 on {metrics['device']}")
    print(f"\nSaved model -> {config.MODEL_PATH}")
    print(f"Saved metrics -> {config.METRICS_PATH}")
    print(f"Saved label map -> {config.LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()
