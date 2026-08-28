"""Export the trained model in smaller formats and report the sizes.

The server runs the full fp32 checkpoint — it has a GPU and no reason to give
up precision. These exports exist for two other purposes:

  * the research report, which claims a compact model and should be able to
    show what "compact" actually costs in accuracy,
  * keeping the door open to running the model somewhere without a GPU, or
    eventually inside the browser.

Every export is re-evaluated on the same held-out test set as train.py, so the
size/accuracy trade-off is measured rather than assumed.

Usage:
    python export_model.py
"""
import copy
import json
import os

import torch
from torch.utils.data import DataLoader

import config
from dataset import SequenceDataset, build_splits
from model import SignCNNTransformer, count_parameters
from train import evaluate_test_set, get_device

EXPORT_DIR = os.path.join(config.SAVED_MODEL_DIR, "export")


def size_mb(path):
    return os.path.getsize(path) / 1e6


def main():
    if not os.path.exists(config.MODEL_PATH):
        raise SystemExit("No trained model found — run `python train.py` first.")

    os.makedirs(EXPORT_DIR, exist_ok=True)
    device = get_device()

    _, _, test_samples, label_map, _, _ = build_splits()
    test_loader = DataLoader(
        SequenceDataset(test_samples, augment=False),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0,
    )

    checkpoint = torch.load(config.MODEL_PATH, map_location="cpu")
    model = SignCNNTransformer(
        num_classes=checkpoint["num_classes"],
        backbone=checkpoint.get("backbone"),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    n_params = count_parameters(model)
    print(f"{checkpoint.get('backbone')}: {n_params/1e6:.2f}M parameters, "
          f"{checkpoint['num_classes']} classes\n")

    results = []

    acc, *_ = evaluate_test_set(copy.deepcopy(model).to(device), test_loader, device)
    results.append(("fp32 (dipakai server)", size_mb(config.MODEL_PATH), acc))

    # half precision: the obvious win, and lossless enough to be worth reporting
    half_path = os.path.join(EXPORT_DIR, "model_fp16.pt")
    half = copy.deepcopy(model).half()
    torch.save({
        "model_state": half.state_dict(),
        "num_classes": checkpoint["num_classes"],
        "backbone": checkpoint.get("backbone"),
        "dtype": "float16",
    }, half_path)

    if device.type == "cuda":
        half_eval = copy.deepcopy(model).to(device).half()

        class _AutoCastWrapper(torch.nn.Module):
            """Feed fp32 batches to the fp16 weights without touching the loader."""

            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, frames, landmarks):
                return self.inner(frames.half(), landmarks.half()).float()

        acc_half, *_ = evaluate_test_set(_AutoCastWrapper(half_eval), test_loader, device)
    else:
        acc_half = float("nan")
    results.append(("fp16", size_mb(half_path), acc_half))

    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)
    metrics["export_sizes_mb"] = {name: round(mb, 2) for name, mb, _ in results}
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"{'format':<26}{'size':>10}{'test acc':>12}")
    for name, mb, acc in results:
        print(f"{name:<26}{mb:>9.2f}M{acc*100:>11.2f}%")
    print(f"\nWritten to {EXPORT_DIR}")
    print("Sizes recorded in metrics.json under 'export_sizes_mb'.")


if __name__ == "__main__":
    main()
