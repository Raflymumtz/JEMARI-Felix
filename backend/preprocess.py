"""Offline dataset preprocessing.

Walks Dataset/DATASET FELIX/<LETTER>/*, keeps only genuine source images,
detects the hand with MediaPipe, crops a square region around it (with
margin), resizes to CACHE_IMG_SIZE and writes the result to
Dataset/PROCESSED/<LETTER>/<index>.jpg together with a meta.npz holding the
hand landmarks and the capture-session key of every frame.

Three filters decide what survives, in this order:

  1. `is_derived()`  — offline-augmented copies (flip*/rotate*/…*_aug*) are
     dropped; augmentation is done online during training instead, where it
     can never leak a transformed training image into the test set.
  2. content hash    — byte-identical duplicates are dropped (class L alone
     ships 172 exact copies named `L_12(1).JPEG` next to `L_12.JPEG`).
  3. MediaPipe       — frames with no detectable hand still get a centre crop
     and a "blind" landmark vector; they stay in the dataset because the
     server produces the same thing when detection drops out.

Crops are cached at CACHE_IMG_SIZE (160) rather than the model's IMG_SIZE
(112) so training-time random-resized-crop has real pixels to zoom into.

Running this once means training and the real-time server share *exactly*
the same crop/landmark logic (see backend/hand_crop.py) without having to
re-run hand detection on every epoch.

Usage:
    python preprocess.py                 # only classes that changed
    python preprocess.py --force         # redo everything
    python preprocess.py --only L,N,P    # redo specific classes
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

import config
from dataset_index import dhash_image, is_derived, natural_sort_key, source_group
from hand_crop import HandCropper

META_FILENAME = "meta.npz"
DHASH_SIZE = 8


def frame_dhash(jpeg_path):
    """Perceptual hash of a written frame, cached so dataset.py can keep
    near-duplicates together without re-reading 13k images every run.

    Hashed from the file *after* JPEG encoding, not from the in-memory crop:
    quality-95 compression shifts a few pixels, which is enough to move a
    borderline pair across the distance threshold and let one look-alike slip
    into the test split. Hashing what actually lands on disk keeps this array
    identical to what audit_dataset.py recomputes.
    """
    gray = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(gray, (DHASH_SIZE + 1, DHASH_SIZE), interpolation=cv2.INTER_AREA)
    return dhash_image(small)


def list_classes():
    return sorted(
        d for d in os.listdir(config.RAW_DATASET_DIR)
        if os.path.isdir(os.path.join(config.RAW_DATASET_DIR, d))
    )


def source_files(letter: str):
    """Raw files of a class in capture order, minus offline-augmented copies."""
    src_dir = os.path.join(config.RAW_DATASET_DIR, letter)
    names = sorted(os.listdir(src_dir), key=natural_sort_key)
    return [n for n in names if not is_derived(os.path.splitext(n)[0])]


def needs_rebuild(letter: str) -> bool:
    """True unless a previous run already covered exactly these source files."""
    meta_path = os.path.join(config.PROCESSED_DATASET_DIR, letter, META_FILENAME)
    if not os.path.exists(meta_path):
        return True
    try:
        with np.load(meta_path, allow_pickle=False) as meta:
            if int(meta["n_source_files"]) != len(source_files(letter)):
                return True
            return int(meta["cache_img_size"]) != config.CACHE_IMG_SIZE
    except Exception:
        return True


def process_class(letter: str):
    src_dir = os.path.join(config.RAW_DATASET_DIR, letter)
    dst_dir = os.path.join(config.PROCESSED_DATASET_DIR, letter)

    # The cache format (resolution, meta.npz, which files are included) changed,
    # so stale frames from an earlier run must not survive alongside new ones.
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    names = source_files(letter)
    cropper = HandCropper(static_image_mode=True)

    seen_hashes = set()
    out_files, out_groups, out_hand, out_landmarks, out_dhash = [], [], [], [], []
    n_hand = n_fallback = n_failed = n_duplicate = 0

    for fname in names:
        src_path = os.path.join(src_dir, fname)
        try:
            with open(src_path, "rb") as fh:
                raw_bytes = fh.read()
        except OSError:
            n_failed += 1
            continue

        digest = hashlib.md5(raw_bytes).hexdigest()
        if digest in seen_hashes:
            n_duplicate += 1
            continue
        seen_hashes.add(digest)

        img = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            n_failed += 1
            continue

        crop, hand_found, landmarks = cropper.crop(img, out_size=config.CACHE_IMG_SIZE)
        if hand_found:
            n_hand += 1
        else:
            n_fallback += 1

        out_name = f"{len(out_files):05d}.jpg"
        out_path = os.path.join(dst_dir, out_name)
        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        out_files.append(out_name)
        out_groups.append(source_group(os.path.splitext(fname)[0]))
        out_hand.append(hand_found)
        out_landmarks.append(landmarks)
        out_dhash.append(frame_dhash(out_path))

    cropper.close()

    np.savez_compressed(
        os.path.join(dst_dir, META_FILENAME),
        files=np.array(out_files),
        group=np.array(out_groups),
        hand_found=np.array(out_hand, dtype=bool),
        landmarks=np.array(out_landmarks, dtype=np.float32).reshape(-1, config.LANDMARK_DIM),
        dhash=np.array(out_dhash, dtype=np.uint8).reshape(-1, DHASH_SIZE),
        n_source_files=np.array(len(names)),
        cache_img_size=np.array(config.CACHE_IMG_SIZE),
    )

    return {
        "letter": letter,
        "raw_total": len(os.listdir(src_dir)),
        "after_derived_filter": len(names),
        "exact_duplicates": n_duplicate,
        "kept": len(out_files),
        "hand_detected": n_hand,
        "fallback_full_frame": n_fallback,
        "failed_to_read": n_failed,
        "groups": len(set(out_groups)),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess the BISINDO dataset.")
    parser.add_argument("--force", action="store_true", help="rebuild every class")
    parser.add_argument("--only", default=None, help="comma-separated classes, e.g. L,N,P")
    args = parser.parse_args()

    os.makedirs(config.PROCESSED_DATASET_DIR, exist_ok=True)
    classes = list_classes()
    print(f"Found {len(classes)} classes: {classes}")

    if args.only:
        wanted = {c.strip().upper() for c in args.only.split(",") if c.strip()}
        todo = [c for c in classes if c.upper() in wanted]
    elif args.force:
        todo = classes
    else:
        todo = [c for c in classes if needs_rebuild(c)]

    skipped = [c for c in classes if c not in todo]
    if skipped:
        print(f"Up to date, skipping {len(skipped)}: {skipped}")
    if not todo:
        print("Nothing to do. Use --force to rebuild anyway.")
        return
    print(f"Processing {len(todo)}: {todo}")

    manifest_path = os.path.join(config.PROCESSED_DATASET_DIR, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            manifest = {}
    # entries for classes we are about to rebuild are replaced wholesale
    manifest = {k: v for k, v in manifest.items() if k in classes and k not in todo}

    t0 = time.time()
    max_workers = min(len(todo), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_class, letter): letter for letter in todo}
        for fut in as_completed(futures):
            stats = fut.result()
            letter = stats.pop("letter")
            manifest[letter] = stats
            print(
                f"[{letter}] raw={stats['raw_total']} "
                f"derived_dropped={stats['raw_total'] - stats['after_derived_filter']} "
                f"dupes={stats['exact_duplicates']} kept={stats['kept']} "
                f"hand={stats['hand_detected']} fallback={stats['fallback_full_frame']} "
                f"groups={stats['groups']}"
            )

    manifest = {k: manifest[k] for k in sorted(manifest)}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    kept = sum(v["kept"] for v in manifest.values())
    hand = sum(v["hand_detected"] for v in manifest.values())
    dropped = sum(v["raw_total"] - v["after_derived_filter"] for v in manifest.values())
    dupes = sum(v["exact_duplicates"] for v in manifest.values())
    dt = time.time() - t0
    print(
        f"\nDone in {dt:.1f}s. {kept} usable frames across {len(manifest)} classes "
        f"({hand} with a detected hand, {hand / max(kept, 1) * 100:.1f}%).\n"
        f"Excluded {dropped} offline-augmented copies and {dupes} exact duplicates.\n"
        f"Manifest written to {manifest_path}"
    )


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
