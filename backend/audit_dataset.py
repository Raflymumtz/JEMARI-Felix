"""Dataset audit — the gate that must pass before training is worth running.

The first version of this project reported 96.99% test accuracy, which turned
out to be an artefact: the raw folders mixed offline-augmented copies
(flip*/rotate*/…_aug*) in with the originals, and the old filename-ordered
split pushed most of those copies into the test block. The model was being
scored on mirrored and rotated versions of its own training images.

preprocess.py now excludes those by filename and by content hash, and
dataset.py splits each capture session separately. This script verifies that
the result is actually clean, by comparing every processed frame against every
other with a perceptual hash (dHash) and reporting any near-duplicate pair
that ended up on opposite sides of a split boundary.

A "cross-split duplicates: 0" line here is the evidence that the accuracy
figure produced by train.py means something.

Usage:
    python audit_dataset.py
    python audit_dataset.py --threshold 5 --show 20
"""
import argparse
import json
import os

import cv2
import numpy as np

import config
from dataset import build_splits, list_classes, load_class_meta, split_class_indices
from dataset_index import dhash_image, hamming_block


def dhash(path, size=8):
    """64-bit difference hash, recomputed from the image on disk.

    Deliberately independent of the dhash cached in meta.npz: this check is
    supposed to catch the split being wrong, so it must not trust the same
    array the splitter used.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    return dhash_image(small)


def find_cross_split_pairs(hashes, split_of, labels, threshold, block=256):
    """Near-duplicate pairs whose two frames live in different splits."""
    n = len(hashes)
    pairs = []
    for start in range(0, n, block):
        stop = min(start + block, n)
        dist = hamming_block(hashes[start:stop], hashes)
        for local_i, i in enumerate(range(start, stop)):
            close = np.nonzero(dist[local_i] <= threshold)[0]
            for j in close:
                if j <= i:
                    continue
                if split_of[i] != split_of[j]:
                    pairs.append((i, int(j), int(dist[local_i, j])))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Audit dataset hygiene and split integrity.")
    parser.add_argument("--threshold", type=int, default=5,
                        help="max Hamming distance counted as a near-duplicate (default 5)")
    parser.add_argument("--show", type=int, default=15, help="how many offending pairs to print")
    args = parser.parse_args()

    classes = list_classes()
    if not classes:
        raise SystemExit("No processed classes found - run `python preprocess.py --force` first.")

    manifest_path = os.path.join(config.PROCESSED_DATASET_DIR, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    print("=" * 78)
    print("EXCLUSIONS (from preprocess.py)")
    print("=" * 78)
    if manifest:
        raw = sum(v["raw_total"] for v in manifest.values())
        derived = sum(v["raw_total"] - v["after_derived_filter"] for v in manifest.values())
        dupes = sum(v["exact_duplicates"] for v in manifest.values())
        kept = sum(v["kept"] for v in manifest.values())
        hand = sum(v["hand_detected"] for v in manifest.values())
        print(f"  raw files                    {raw:>7}")
        print(f"  offline-augmented copies     {derived:>7}  ({derived/raw*100:.1f}%)")
        print(f"  byte-identical duplicates    {dupes:>7}")
        print(f"  frames kept                  {kept:>7}")
        print(f"  ... with a detected hand     {hand:>7}  ({hand/max(kept,1)*100:.1f}%)")
    else:
        print("  manifest.json missing - rerun preprocess.py")

    print()
    print("=" * 78)
    print("SPLIT LAYOUT (per capture session, chronological, WINDOW-frame embargo)")
    print("=" * 78)
    print(f"{'cls':<5}{'frames':>7}{'sess':>6}{'train':>8}{'val':>7}{'test':>7}{'quar':>6}   sessions")

    all_hashes, all_split, all_label, all_path = [], [], [], []
    total_quarantined = 0
    for c in classes:
        meta = load_class_meta(c)
        groups, n_demoted = split_class_indices(meta, return_stats=True)
        total_quarantined += n_demoted
        n_tr = sum(len(t) for _, t, _, _ in groups)
        n_va = sum(len(v) for _, _, v, _ in groups)
        n_te = sum(len(t) for _, _, _, t in groups)
        names = ", ".join(g for g, _, _, _ in groups[:4])
        print(f"{c:<5}{len(meta['files']):>7}{len(groups):>6}{n_tr:>8}{n_va:>7}{n_te:>7}"
              f"{n_demoted:>6}   {names}")

        split_of = {}
        for _, tr, va, te in groups:
            for i in tr:
                split_of[i] = "train"
            for i in va:
                split_of[i] = "val"
            for i in te:
                split_of[i] = "test"

        for i, fname in enumerate(meta["files"]):
            if i not in split_of:
                continue  # dropped by the embargo gap
            path = os.path.join(meta["dir"], fname)
            h = dhash(path)
            if h is None:
                continue
            all_hashes.append(h)
            all_split.append(split_of[i])
            all_label.append(c)
            all_path.append(path)

    hashes = np.stack(all_hashes)
    print(f"\nHashed {len(hashes)} frames (frames inside the embargo gap are excluded from training entirely).")
    print(f"Quarantined into train because a look-alike sat in another split: {total_quarantined}")

    print()
    print("=" * 78)
    print(f"NEAR-DUPLICATE LEAK CHECK (dHash, Hamming <= {args.threshold})")
    print("=" * 78)

    pairs = find_cross_split_pairs(hashes, all_split, all_label, args.threshold)
    same_class = [p for p in pairs if all_label[p[0]] == all_label[p[1]]]
    cross_class = [p for p in pairs if all_label[p[0]] != all_label[p[1]]]

    print(f"  cross-split duplicates (same letter, real leakage):  {len(same_class)}")
    print(f"  cross-split duplicates (different letter, label noise): {len(cross_class)}")

    for i, j, d in same_class[:args.show]:
        print(f"    LEAK [{all_label[i]}] dist={d}  {all_split[i]}:{os.path.basename(all_path[i])}"
              f"  <->  {all_split[j]}:{os.path.basename(all_path[j])}")
    for i, j, d in cross_class[:args.show]:
        print(f"    NOISE dist={d}  {all_label[i]}:{os.path.basename(all_path[i])}"
              f"  <->  {all_label[j]}:{os.path.basename(all_path[j])}")

    print()
    print("=" * 78)
    print("SEQUENCE WINDOWS")
    print("=" * 78)
    train_s, val_s, test_s, label_map, _, _ = build_splits()
    print(f"  classes {len(label_map)}  train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    print()
    if same_class:
        print(f"FAIL: {len(same_class)} near-duplicate pairs still straddle a split boundary.")
        print("      Test accuracy would be inflated. Fix before training.")
        raise SystemExit(1)
    print("PASS: no near-duplicate frame is shared between splits.")
    print("      Test-set accuracy from train.py can be reported as-is.")


if __name__ == "__main__":
    main()
