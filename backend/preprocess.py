"""Offline dataset preprocessing.

Walks Dataset/DATASET FELIX/<LETTER>/*.jpg, detects the hand with MediaPipe
Hands, crops a square region around it (with margin), resizes to IMG_SIZE
and writes the result to Dataset/PROCESSED/<LETTER>/<index>.jpg.

Running this once means training and the real-time server share *exactly*
the same crop/resize logic (see backend/hand_crop.py) without having to
re-run hand detection on every epoch.

Usage:
    python preprocess.py
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

import config
from hand_crop import HandCropper

_NUM_RE = re.compile(r"(\d+)")


def natural_sort_key(name: str):
    parts = _NUM_RE.split(name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_classes():
    return sorted(
        d for d in os.listdir(config.RAW_DATASET_DIR)
        if os.path.isdir(os.path.join(config.RAW_DATASET_DIR, d))
    )


def process_class(letter: str):
    src_dir = os.path.join(config.RAW_DATASET_DIR, letter)
    dst_dir = os.path.join(config.PROCESSED_DATASET_DIR, letter)
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(os.listdir(src_dir), key=natural_sort_key)
    cropper = HandCropper(static_image_mode=True)

    n_hand, n_fallback, n_failed = 0, 0, 0
    for i, fname in enumerate(files):
        src_path = os.path.join(src_dir, fname)
        img = cv2.imread(src_path)
        if img is None:
            n_failed += 1
            continue
        crop, hand_found = cropper.crop(img)
        if hand_found:
            n_hand += 1
        else:
            n_fallback += 1
        out_path = os.path.join(dst_dir, f"{i:05d}.jpg")
        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    cropper.close()
    return letter, len(files), n_hand, n_fallback, n_failed


def main():
    os.makedirs(config.PROCESSED_DATASET_DIR, exist_ok=True)
    classes = list_classes()
    print(f"Found {len(classes)} classes: {classes}")

    manifest = {}
    t0 = time.time()
    max_workers = min(len(classes), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_class, letter): letter for letter in classes}
        for fut in as_completed(futures):
            letter, total, n_hand, n_fallback, n_failed = fut.result()
            manifest[letter] = {
                "total": total,
                "hand_detected": n_hand,
                "fallback_full_frame": n_fallback,
                "failed_to_read": n_failed,
            }
            print(
                f"[{letter}] total={total} hand_detected={n_hand} "
                f"fallback={n_fallback} failed={n_failed}"
            )

    with open(os.path.join(config.PROCESSED_DATASET_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    dt = time.time() - t0
    print(f"Done in {dt:.1f}s. Manifest written to Dataset/PROCESSED/manifest.json")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
