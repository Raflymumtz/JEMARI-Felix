"""Builds windowed (spatio-temporal) sequence samples from the pre-cropped
BISINDO frame dataset and exposes a PyTorch Dataset + train/val/test split.

Each class folder under Dataset/PROCESSED/<LETTER>/ holds frames captured
in chronological order. To avoid leaking near-duplicate frames between
splits, the file list of every class is first cut chronologically into a
70/15/15 train/val/test block, and only *then* is a sliding window
(config.WINDOW, config.STRIDE) applied within each block. No window ever
spans a split boundary.
"""
import json
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

import config

_NUM_RE = re.compile(r"(\d+)")


def natural_sort_key(name: str):
    parts = _NUM_RE.split(name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_classes():
    classes = sorted(
        d for d in os.listdir(config.PROCESSED_DATASET_DIR)
        if os.path.isdir(os.path.join(config.PROCESSED_DATASET_DIR, d))
    )
    return classes


def _make_windows(file_paths, label, window, stride):
    samples = []
    n = len(file_paths)
    if n < window:
        if n > 0:
            # pad by repeating the last frame so short splits are still usable
            padded = file_paths + [file_paths[-1]] * (window - n)
            samples.append((padded, label))
        return samples
    for start in range(0, n - window + 1, stride):
        samples.append((file_paths[start:start + window], label))
    # make sure the tail of the sequence is represented too
    if (n - window) % stride != 0:
        samples.append((file_paths[n - window:n], label))
    return samples


def build_splits():
    classes = list_classes()
    label_map = {i: c for i, c in enumerate(classes)}
    class_to_idx = {c: i for i, c in label_map.items()}

    train_samples, val_samples, test_samples = [], [], []
    class_counts = {c: 0 for c in classes}

    for c in classes:
        class_dir = os.path.join(config.PROCESSED_DATASET_DIR, c)
        files = sorted(os.listdir(class_dir), key=natural_sort_key)
        files = [os.path.join(class_dir, f) for f in files]
        class_counts[c] = len(files)

        n = len(files)
        n_train = int(n * (1 - config.VAL_SPLIT - config.TEST_SPLIT))
        n_val = int(n * config.VAL_SPLIT)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        label = class_to_idx[c]
        train_samples += _make_windows(train_files, label, config.WINDOW, config.STRIDE)
        val_samples += _make_windows(val_files, label, config.WINDOW, config.WINDOW)
        test_samples += _make_windows(test_files, label, config.WINDOW, config.WINDOW)

    return train_samples, val_samples, test_samples, label_map, class_counts


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SequenceDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, path, flip):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if flip:
            img = cv2.flip(img, 1)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(img, (2, 0, 1))  # CHW

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        flip = self.augment and np.random.rand() < 0.5
        frames = np.stack([self._load_frame(p, flip) for p in paths], axis=0)  # T,C,H,W
        return torch.from_numpy(frames), label


def compute_class_weights(train_samples, num_classes):
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, label in train_samples:
        counts[label] += 1
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    train_samples, val_samples, test_samples, label_map, class_counts = build_splits()
    print("classes:", label_map)
    print("raw frame counts per class:", class_counts)
    print(f"train windows={len(train_samples)} val windows={len(val_samples)} test windows={len(test_samples)}")
