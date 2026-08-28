"""GPU-accelerated training script for the BISINDO CNN+Transformer model.

Implements BAB 3.3 / 3.4 of the proposal:
  - loads the pre-cropped dataset (see preprocess.py)
  - splits it 70% train / 15% validation / 15% test, cut per capture session
    with an embargo gap so no near-duplicate frame leaks across splits
    (see dataset.build_splits)
  - trains a MobileNet CNN + Transformer (self-attention) end to end, with an
    auxiliary hand-landmark stream
  - reports accuracy, precision, recall, F1-score and inference latency on
    the held-out test set, exactly the metrics named in the proposal, plus a
    per-class breakdown and a confusion matrix.

Run:
    python train.py
Requires an NVIDIA GPU (VGA) with CUDA; falls back to CPU automatically
if none is found, but that will be far slower.
"""
import argparse
import copy
import csv
import gc
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config
from dataset import (
    SequenceDataset,
    build_splits,
    compute_class_weights,
    compute_sample_weights,
    normalize_frames,
)
from model import SignCNNTransformer, count_parameters
from report import print_report


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
    print("[CPU] No CUDA GPU found - training on CPU (this will be slow).")
    return torch.device("cpu")


class ModelEMA:
    """Exponential moving average of the weights.

    The averaged weights are noticeably steadier than the raw ones on a
    dataset this size, so evaluation and the saved checkpoint both use them.
    """

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # Ramp the decay in rather than starting at 0.999. There are only ~124
        # steps per epoch here, so a fixed 0.999 (an effective window of ~1000
        # steps) would leave the average still dominated by the random
        # initialisation for the first eight epochs — wasting them, and making
        # early validation numbers meaningless in the training curve.
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for ema_v, model_v in zip(self.module.state_dict().values(),
                                  model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(decay).add_(model_v.detach(), alpha=1.0 - decay)
            else:
                ema_v.copy_(model_v)


def build_scheduler(optimizer):
    """Linear warmup then cosine decay, per epoch."""
    def lr_lambda(epoch):
        if epoch < config.WARMUP_EPOCHS:
            return (epoch + 1) / max(config.WARMUP_EPOCHS, 1)
        progress = (epoch - config.WARMUP_EPOCHS) / max(config.EPOCHS - config.WARMUP_EPOCHS, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(model):
    """Lower LR for the pretrained backbone than for the parts trained from scratch."""
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone_params if name.startswith("cnn.") else head_params).append(param)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.LR_BACKBONE},
            {"params": head_params, "lr": config.LR_HEAD},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )


def run_epoch(model, loader, criterion, optimizer, device, scaler, ema=None, train=True):
    model.train(mode=train)
    total_loss, all_preds, all_labels = 0.0, [], []

    for frames, landmarks, labels in loader:
        # frames arrive as uint8 and are normalised here, on the GPU
        frames = normalize_frames(frames.to(device, non_blocking=True))
        landmarks = landmarks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(frames, landmarks)
                loss = criterion(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)

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
def evaluate_test_set(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for frames, landmarks, labels in loader:
        frames = normalize_frames(frames.to(device, non_blocking=True))
        landmarks = landmarks.to(device, non_blocking=True)
        logits = model(frames, landmarks)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return acc, precision, recall, f1, all_labels, all_preds


def weighted_metrics(y_true, y_pred):
    """Support-weighted averages, reported next to the macro ones.

    Macro treats every letter equally, which is the fairer figure when the
    letters have unequal test counts; weighted reflects what a user actually
    experiences on this test set. Quoting both avoids implying either one is
    "the" accuracy.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return float(precision), float(recall), float(f1)


@torch.no_grad()
def measure_latency(model, device, n_warmup=10, n_measure=50):
    model.eval()
    frames = torch.randn(1, config.WINDOW, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)
    landmarks = torch.randn(1, config.WINDOW, config.LANDMARK_DIM, device=device)

    for _ in range(n_warmup):
        _ = model(frames, landmarks)
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        _ = model(frames, landmarks)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(timings)), float(np.percentile(timings, 95))


def per_class_metrics(y_true, y_pred, label_map):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=sorted(label_map), zero_division=0
    )
    return {
        label_map[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in sorted(label_map)
    }


def dataset_hygiene():
    """Audit trail of what preprocess.py excluded, for the research report."""
    manifest_path = os.path.join(config.PROCESSED_DATASET_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {
        "raw_files": sum(v["raw_total"] for v in manifest.values()),
        "offline_augmented_excluded": sum(
            v["raw_total"] - v["after_derived_filter"] for v in manifest.values()
        ),
        "exact_duplicates_excluded": sum(v["exact_duplicates"] for v in manifest.values()),
        "frames_used": sum(v["kept"] for v in manifest.values()),
        "frames_with_detected_hand": sum(v["hand_detected"] for v in manifest.values()),
        "capture_sessions": {k: v["groups"] for k, v in manifest.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Train the BISINDO CNN+Transformer model.")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training: evaluate the saved checkpoint and write the report")
    args = parser.parse_args()

    set_seed(config.SEED)
    device = get_device()
    # every batch has the same shape, so autotuning pays off immediately
    torch.backends.cudnn.benchmark = True
    os.makedirs(config.SAVED_MODEL_DIR, exist_ok=True)

    print("Building group-aware train/val/test window splits...")
    train_samples, val_samples, test_samples, label_map, class_counts, group_counts = build_splits()
    num_classes = len(label_map)
    print(f"Classes ({num_classes}): {list(label_map.values())}")
    print(f"Frames per class: {class_counts}")
    print(f"Sequence windows -> train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    train_ds = SequenceDataset(train_samples, augment=True)
    val_ds = SequenceDataset(val_samples, augment=False)
    test_ds = SequenceDataset(test_samples, augment=False)

    # Only the *training* loader gets worker processes. It is the one doing
    # heavy online augmentation over 4551 windows; validation and test are
    # ~230 windows each with no augmentation, which a single process finishes
    # in well under a second.
    #
    # This is not just a tuning choice. On Windows every spawned worker
    # imports torch and commits 1-2 GB of paging file for the CUDA DLLs.
    # Giving all three loaders 12 persistent workers meant 24 processes alive
    # through training and 36 once the test loader started, which failed with
    # "WinError 1455: The paging file is too small for this operation" —
    # after 21 epochs of otherwise perfect training.
    num_workers = min(12, max(1, (os.cpu_count() or 4) // 2))
    train_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    eval_kwargs = dict(num_workers=0, pin_memory=(device.type == "cuda"))

    sampler = WeightedRandomSampler(
        compute_sample_weights(train_samples, num_classes),
        num_samples=len(train_samples),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler, **train_kwargs)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, **eval_kwargs)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, **eval_kwargs)

    # Written before training rather than at the end. The checkpoint is saved
    # every time validation improves, so a run interrupted midway would
    # otherwise leave a new model.pt beside the *previous* run's label map —
    # and the server would happily load both and mislabel every letter.
    with open(config.LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    model = SignCNNTransformer(num_classes=num_classes,
                               pretrained=not args.eval_only).to(device)
    n_params = count_parameters(model)
    print(f"Backbone {model.backbone_name}: {n_params/1e6:.2f}M parameters")

    ema = ModelEMA(model, config.EMA_DECAY)

    class_weights = compute_class_weights(train_samples, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.LABEL_SMOOTHING)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    best_val_f1 = -1.0
    patience_counter = 0
    history = []

    for epoch in range(1, 1 if args.eval_only else config.EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, ema=ema, train=True
        )
        val_loss, val_acc, val_f1 = run_epoch(
            ema.module, val_loader, criterion, optimizer, device, scaler, train=False
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
            torch.save({
                "model_state": ema.module.state_dict(),
                "num_classes": num_classes,
                "backbone": model.backbone_name,
            }, config.MODEL_PATH)
            print(f"  -> new best val_f1={val_f1:.4f}, checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping (no val_f1 improvement in {config.PATIENCE} epochs).")
                break

    # Empty under --eval-only; the existing curve from the real run is kept.
    if history:
        with open(config.HISTORY_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    # Training is over, so hand the worker processes — and the paging file they
    # reserved — back before the evaluation pass allocates anything further.
    del train_loader, val_loader
    gc.collect()

    print("\nLoading best checkpoint for final test-set evaluation...")
    if not os.path.exists(config.MODEL_PATH):
        raise SystemExit(f"{config.MODEL_PATH} not found - run training first.")
    checkpoint = torch.load(config.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    acc, precision, recall, f1, y_true, y_pred = evaluate_test_set(model, test_loader, device)
    latency_mean, latency_p95 = measure_latency(model, device)
    cpu_latency_mean, _ = measure_latency(copy.deepcopy(model).to("cpu"), torch.device("cpu"),
                                          n_warmup=3, n_measure=15)

    w_precision, w_recall, w_f1 = weighted_metrics(y_true, y_pred)

    metrics = {
        "test_accuracy": acc,
        "test_error_rate": 1.0 - acc,
        "test_precision_macro": precision,
        "test_recall_macro": recall,
        "test_f1_macro": f1,
        "test_precision_weighted": w_precision,
        "test_recall_weighted": w_recall,
        "test_f1_weighted": w_f1,
        "latency_ms_mean": latency_mean,
        "latency_ms_p95": latency_p95,
        "latency_ms_mean_cpu": cpu_latency_mean,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "backbone": model.backbone_name,
        "n_params": n_params,
        "model_size_mb": os.path.getsize(config.MODEL_PATH) / 1e6,
        "num_classes": num_classes,
        "window": config.WINDOW,
        "img_size": config.IMG_SIZE,
        "train_windows": len(train_samples),
        "val_windows": len(val_samples),
        "test_windows": len(test_samples),
        "raw_frame_counts": class_counts,
        "capture_sessions_per_class": group_counts,
        "per_class": per_class_metrics(y_true, y_pred, label_map),
        "dataset_hygiene": dataset_hygiene(),
    }

    letters = [label_map[i] for i in sorted(label_map)]
    cm = confusion_matrix(y_true, y_pred, labels=sorted(label_map))
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classes"] = letters

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(config.CONFUSION_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + letters)
        for letter, row in zip(letters, cm):
            writer.writerow([letter] + row.tolist())

    # One implementation of the report, shared with `python report.py`, so the
    # numbers shown at the end of training and the ones printed later can
    # never drift apart.
    print_report(metrics, cm, letters)

    print(f"Model tersimpan         -> {config.MODEL_PATH}")
    print(f"Metrik tersimpan        -> {config.METRICS_PATH}")
    print(f"Confusion matrix (CSV)  -> {config.CONFUSION_PATH}")
    print(f"Kurva training (CSV)    -> {config.HISTORY_PATH}")
    print("\nCetak ulang laporan ini kapan saja:  python report.py")


if __name__ == "__main__":
    main()
