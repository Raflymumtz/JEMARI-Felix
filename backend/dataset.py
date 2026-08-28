"""Builds windowed (spatio-temporal) sequence samples from the pre-cropped
BISINDO frame dataset and exposes a PyTorch Dataset + train/val/test split.

Splitting is **group-aware**. Each class folder mixes several capture
sessions (webcam bursts, phone photos, Roboflow exports — see
dataset_index.source_group), and consecutive frames inside one session are
near-identical. So the split is cut per *(class, session)*: every session is
sliced chronologically into a 70/15/15 block, an embargo of `config.WINDOW`
frames is discarded at each boundary, and only then is a sliding window
applied inside each block. No window ever spans a split boundary, a session
boundary, or the embargo gap — while every session still contributes to all
three splits, so train and test see the same range of backgrounds.

Augmentation happens **online** here rather than as files on disk. The raw
dataset shipped ~18.7k pre-augmented copies mixed in with the originals,
which is exactly how transformed training images ended up in the old test
set. Generating them per epoch instead gives far more variety and cannot
leak: the geometric transform is drawn once per window (so the hand does not
teleport between frames of one sequence) while photometric noise is redrawn
per frame (so the Transformer learns to stabilise a jittery webcam stream
instead of eight identical stills).
"""
import json
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config
from dataset_index import cluster_near_duplicates, natural_sort_key

META_FILENAME = "meta.npz"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_classes():
    classes = sorted(
        d for d in os.listdir(config.PROCESSED_DATASET_DIR)
        if os.path.isdir(os.path.join(config.PROCESSED_DATASET_DIR, d))
    )
    return classes


def load_class_meta(letter: str):
    """Frames of one class with their session key, landmarks and hand flag."""
    class_dir = os.path.join(config.PROCESSED_DATASET_DIR, letter)
    meta_path = os.path.join(class_dir, META_FILENAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} missing - run `python preprocess.py --force` first."
        )
    with np.load(meta_path, allow_pickle=False) as meta:
        return {
            "dir": class_dir,
            "files": [str(f) for f in meta["files"]],
            "group": [str(g) for g in meta["group"]],
            "hand_found": meta["hand_found"],
            "landmarks": meta["landmarks"].astype(np.float32),
            "dhash": meta["dhash"],
        }


def _make_windows(indices, window, stride, allow_pad=True):
    """Slice a run of frame indices into overlapping windows.

    `allow_pad` is on for training, where repeating the last frame of a short
    run is harmless extra data, and off for val/test, where a "window" of five
    real frames plus three copies of one of them would be scored as if it were
    eight independent observations.
    """
    samples = []
    n = len(indices)
    if n == 0:
        return samples
    if n < window:
        if not allow_pad:
            return samples
        samples.append(list(indices) + [indices[-1]] * (window - n))
        return samples
    for start in range(0, n - window + 1, stride):
        samples.append(list(indices[start:start + window]))
    # make sure the tail of the sequence is represented too
    if (n - window) % stride != 0:
        samples.append(list(indices[n - window:n]))
    return samples


def _split_group(indices, embargo):
    """Chronological 70/15/15 cut of one capture session, with an embargo gap.

    The gap matters: frame 76 and 77 of a webcam burst are nearly the same
    picture, so a bare cut would still place a near-duplicate pair on opposite
    sides of the train/val boundary.
    """
    n = len(indices)
    n_train = int(n * (1 - config.VAL_SPLIT - config.TEST_SPLIT))
    n_val = int(n * config.VAL_SPLIT)

    train = indices[:max(n_train - embargo, 0)]
    val = indices[n_train:max(n_train + n_val - embargo, n_train)]
    test = indices[n_train + n_val:]
    return train, val, test


def _quarantine_duplicates(groups, dhashes):
    """Force every near-duplicate cluster onto the same side of the split.

    Session grouping alone is not enough: the same photograph appears under
    two naming schemes (a Roboflow re-export of an image that also sits in the
    folder as `c_12.jpg`), so its two copies land in two sessions and then in
    two different splits. An audit of the first attempt found 125 such pairs.

    Whenever a cluster spans more than one split, all of its frames are moved
    into *train*. Moving toward train can only shrink the evaluation set,
    never inflate it, so the reported accuracy stays honest by construction.
    """
    split_of = {}
    for _, tr, va, te in groups:
        for i in tr:
            split_of[i] = "train"
        for i in va:
            split_of[i] = "val"
        for i in te:
            split_of[i] = "test"

    demote = set()
    for cluster in cluster_near_duplicates(dhashes, config.DUP_HAMMING):
        splits = {split_of[i] for i in cluster if i in split_of}
        if len(splits) > 1:
            demote.update(i for i in cluster if i in split_of)

    if not demote:
        return groups, 0

    rebuilt = []
    for g, tr, va, te in groups:
        moved = [i for i in va + te if i in demote]
        rebuilt.append((
            g,
            sorted(set(tr) | set(moved)),
            [i for i in va if i not in demote],
            [i for i in te if i not in demote],
        ))
    return rebuilt, len(demote)


def min_splittable_group():
    """Smallest session that can be cut three ways and still fill every window.

    With a WINDOW-frame embargo at each boundary, a 20-frame session splits
    into 14/0/3 — no training window, no validation, and a test "window" that
    is three real frames padded out with copies. Sessions below this size are
    assigned whole instead.
    """
    n = config.WINDOW * 2
    while n < 10_000:
        n_train = int(n * (1 - config.VAL_SPLIT - config.TEST_SPLIT))
        n_val = int(n * config.VAL_SPLIT)
        n_test = n - n_train - n_val
        if (n_train - config.WINDOW >= config.WINDOW
                and n_val - config.WINDOW >= config.WINDOW
                and n_test >= config.WINDOW):
            return n
        n += 1
    return 10_000


def split_class_indices(meta, return_stats=False):
    """Frame indices of one class, grouped by capture session and split.

    Two regimes, because the sessions differ wildly in size (4 frames for
    `body dot (#)`, 300+ for `a_#`):

      * Large sessions are cut chronologically 70/15/15 with an embargo, so
        every split sees that background/camera/lighting.
      * Small sessions go to a single split, picked greedily to keep the
        class near its overall 70/15/15 frame budget. Splitting them three
        ways instead would yield empty validation sets and padded test
        windows, which is how the previous version ended up scoring 61 of its
        265 sessions on windows containing fewer than eight distinct frames.

    Returns [(group_key, train_idx, val_idx, test_idx), ...]. Kept separate
    from build_splits so audit_dataset.py can check the very same assignment
    that training will use.
    """
    by_group = {}
    for i, g in enumerate(meta["group"]):
        by_group.setdefault(g, []).append(i)

    total = sum(len(v) for v in by_group.values())
    target = {
        "train": total * (1 - config.VAL_SPLIT - config.TEST_SPLIT),
        "val": total * config.VAL_SPLIT,
        "test": total * config.TEST_SPLIT,
    }
    current = {"train": 0, "val": 0, "test": 0}
    min_splittable = min_splittable_group()

    groups = []
    # largest first, so the big sessions set the baseline and the small ones
    # are used to top up whichever split is furthest behind
    for g, indices in sorted(by_group.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        indices.sort(key=lambda i: natural_sort_key(meta["files"][i]))

        if len(indices) >= min_splittable:
            tr, va, te = _split_group(indices, config.WINDOW)
        elif len(indices) < config.WINDOW:
            # cannot fill a single window on its own — only useful for training
            tr, va, te = indices, [], []
        else:
            pick = max(target, key=lambda k: target[k] - current[k])
            tr = indices if pick == "train" else []
            va = indices if pick == "val" else []
            te = indices if pick == "test" else []

        for key, part in (("train", tr), ("val", va), ("test", te)):
            current[key] += len(part)
        groups.append((g, tr, va, te))

    groups.sort(key=lambda t: t[0])
    groups, n_demoted = _quarantine_duplicates(groups, meta["dhash"])
    return (groups, n_demoted) if return_stats else groups


# --------------------------------------------------------------------------
# frame cache
# --------------------------------------------------------------------------
#
# Decoding JPEGs was 30% of the whole data pipeline and, together with the
# rest of the per-frame CPU work, left the GPU idle roughly 80% of the time.
# Every processed frame is therefore decoded exactly once into a single
# memory-mapped uint8 array (~1 GB for 13.7k frames at 160px).
#
# A memory map rather than a shared torch tensor: the operating system's page
# cache backs the file, so all twelve DataLoader workers read the same
# physical pages without any of them holding a private copy, and it survives
# process spawn on Windows without special handling.

FRAME_CACHE_PATH = os.path.join(config.PROCESSED_DATASET_DIR, "frames_cache.npy")
FRAME_CACHE_INDEX = os.path.join(config.PROCESSED_DATASET_DIR, "frames_cache.json")


def _cache_signature(classes):
    return {
        "cache_img_size": config.CACHE_IMG_SIZE,
        "counts": {c: len(load_class_meta(c)["files"]) for c in classes},
    }


def ensure_frame_cache(verbose=True):
    """Build the decoded-frame cache if it is missing or out of date.

    Returns {class: first global frame index}. Takes ~15 s to build and is
    then reused by every later run.
    """
    classes = list_classes()
    signature = _cache_signature(classes)

    if os.path.exists(FRAME_CACHE_PATH) and os.path.exists(FRAME_CACHE_INDEX):
        with open(FRAME_CACHE_INDEX) as f:
            stored = json.load(f)
        if stored.get("signature") == signature:
            return stored["offsets"]

    size = config.CACHE_IMG_SIZE
    offsets, total = {}, 0
    for c in classes:
        offsets[c] = total
        total += signature["counts"][c]

    if verbose:
        print(f"Building frame cache: {total} frames at {size}px "
              f"({total * size * size * 3 / 1e9:.2f} GB) -> {FRAME_CACHE_PATH}")

    arr = np.lib.format.open_memmap(
        FRAME_CACHE_PATH, mode="w+", dtype=np.uint8, shape=(total, size, size, 3)
    )
    for c in classes:
        meta = load_class_meta(c)
        base = offsets[c]
        for i, fname in enumerate(meta["files"]):
            img = cv2.imread(os.path.join(meta["dir"], fname))
            if img.shape[0] != size:
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            arr[base + i] = img
        if verbose:
            print(f"  cached {c} ({len(meta['files'])} frames)", flush=True)
    arr.flush()
    del arr

    with open(FRAME_CACHE_INDEX, "w") as f:
        json.dump({"signature": signature, "offsets": offsets}, f, indent=2)
    if verbose:
        print("Frame cache ready.")
    return offsets


def frame_paths():
    """Global frame index -> path on disk (for tools that need the file)."""
    paths = []
    for c in list_classes():
        meta = load_class_meta(c)
        paths.extend(os.path.join(meta["dir"], f) for f in meta["files"])
    return paths


def build_splits():
    classes = list_classes()
    label_map = {i: c for i, c in enumerate(classes)}
    class_to_idx = {c: i for i, c in label_map.items()}
    offsets = ensure_frame_cache()

    train_samples, val_samples, test_samples = [], [], []
    class_counts = {}
    group_counts = {}

    for c in classes:
        meta = load_class_meta(c)
        label = class_to_idx[c]
        base = offsets[c]
        landmarks = meta["landmarks"]
        class_counts[c] = len(meta["files"])

        groups = split_class_indices(meta)
        group_counts[c] = len(groups)

        for _, tr, va, te in groups:
            for dst, idx_list, stride, allow_pad in (
                (train_samples, tr, config.STRIDE, True),
                (val_samples, va, config.WINDOW, False),
                (test_samples, te, config.WINDOW, False),
            ):
                for window in _make_windows(idx_list, config.WINDOW, stride, allow_pad):
                    dst.append((
                        np.array([base + i for i in window], dtype=np.int64),
                        landmarks[window],
                        label,
                    ))

    return train_samples, val_samples, test_samples, label_map, class_counts, group_counts


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------

def _geometric_params():
    """One geometric transform per window, shared by all frames in it."""
    area = np.random.uniform(0.65, 1.0)
    side = config.CACHE_IMG_SIZE * math.sqrt(area)
    max_shift = (config.CACHE_IMG_SIZE - side) / 2
    center = config.CACHE_IMG_SIZE / 2
    return {
        "cx": center + np.random.uniform(-max_shift, max_shift),
        "cy": center + np.random.uniform(-max_shift, max_shift),
        "side": side,
        "angle": np.random.uniform(-20.0, 20.0),
        "flip": np.random.rand() < 0.5,
    }


def _affine_matrix(p, out_size):
    """Source (cached crop) pixels -> augmented output pixels."""
    rad = math.radians(p["angle"])
    cos, sin = math.cos(rad), math.sin(rad)
    k = out_size / p["side"]
    fx = -1.0 if p["flip"] else 1.0

    m00, m01 = k * cos * fx, k * sin
    m10, m11 = -k * sin * fx, k * cos
    tx = out_size / 2 - (m00 * p["cx"] + m01 * p["cy"])
    ty = out_size / 2 - (m10 * p["cx"] + m11 * p["cy"])
    return np.float32([[m00, m01, tx], [m10, m11, ty]])


def apply_geometry_to_landmarks(p, landmarks):
    """Rotate/flip the landmark descriptor to match the image transform.

    Cached landmarks are already wrist-centred and scale-normalised (see
    hand_crop.normalize_landmarks), so translation and zoom are no-ops here —
    only the rotation and the mirror actually move the joints. The linear part
    used below is exactly the image matrix with the scale factor `k` divided
    out, which keeps pixels and joints in lockstep.
    """
    rad = math.radians(p["angle"])
    cos, sin = math.cos(rad), math.sin(rad)
    fx = -1.0 if p["flip"] else 1.0

    out = landmarks.copy()
    coords = out[:, :-1].reshape(len(out), config.N_LANDMARKS, 3)
    x, y = coords[..., 0].copy(), coords[..., 1].copy()
    coords[..., 0] = cos * fx * x + sin * y
    coords[..., 1] = -sin * fx * x + cos * y
    # z is a rotation about the image plane's normal, so depth is unchanged
    return out


def _photometric(img):
    """Per-frame camera-like noise. Never touches geometry, so landmarks stay valid.

    Everything stays in uint8. The first version converted to float32 up front
    and back again for each OpenCV call, which cost 5.5 dtype conversions per
    frame — 15% of the entire data pipeline — for no benefit, since every
    operation here is a saturating 8-bit one anyway.
    """
    # brightness / contrast, saturating, in a single C call.
    # addWeighted and not convertScaleAbs: the latter takes the absolute value,
    # so a dark pixel pushed below zero comes back *brighter* instead of
    # clipping to black — a solarisation artefact that never occurs in a real
    # camera and that the model would have to learn to ignore.
    img = cv2.addWeighted(img, np.random.uniform(0.75, 1.3), img, 0,
                          np.random.uniform(-25, 25))

    # saturation / hue
    if np.random.rand() < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = hsv[..., 0].astype(np.int16) + int(np.random.uniform(-8, 8))
        hsv[..., 0] = np.mod(hue, 180).astype(np.uint8)
        sat = hsv[..., 1]
        hsv[..., 1] = cv2.addWeighted(sat, np.random.uniform(0.7, 1.3), sat, 0, 0)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if np.random.rand() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), np.random.uniform(0.3, 1.2))

    # webcam frames reach the server as JPEG, so train on that artefact too
    if np.random.rand() < 0.3:
        quality = int(np.random.uniform(30, 80))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return img


def _random_erase(img):
    h, w = img.shape[:2]
    eh, ew = int(h * np.random.uniform(0.1, 0.3)), int(w * np.random.uniform(0.1, 0.3))
    y0 = np.random.randint(0, max(h - eh, 1))
    x0 = np.random.randint(0, max(w - ew, 1))
    img[y0:y0 + eh, x0:x0 + ew] = np.random.randint(0, 256)
    return img


# ImageNet statistics as tensors, shaped to broadcast over (B, T, C, H, W).
_MEAN_T = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
_STD_T = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)


def normalize_frames(frames_uint8):
    """uint8 (..., C, H, W) -> normalised float32, on whatever device it is on.

    Batches travel from the DataLoader workers to the training process as
    uint8. As float32 a single batch is 38 MB, and at 143 batches per epoch
    that was ~5.5 GB per epoch pushed through the multiprocessing pipe — which
    is what left the GPU idle 80% of the time. uint8 quarters the traffic and
    the conversion below happens on the GPU, where it is free.
    """
    mean = _MEAN_T.to(frames_uint8.device)
    std = _STD_T.to(frames_uint8.device)
    x = frames_uint8.float().div_(255.0)
    if x.dim() == 4:            # (T, C, H, W) — single sequence
        return (x - mean[0]) / std[0]
    return (x - mean) / std


class SequenceDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
        self._frames = None

    def __len__(self):
        return len(self.samples)

    @property
    def frames(self):
        """The shared frame cache, opened lazily once per worker process.

        Opened here rather than in __init__ so the memory map is never
        pickled across the process boundary — each worker maps the same file
        itself and the OS page cache does the sharing.
        """
        if self._frames is None:
            self._frames = np.load(FRAME_CACHE_PATH, mmap_mode="r")
        return self._frames

    def _to_chw(self, img_bgr):
        """BGR HWC uint8 -> RGB CHW uint8. Normalisation happens on the GPU."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))

    def __getitem__(self, idx):
        frame_ids, landmarks, label = self.samples[idx]

        if self.augment:
            params = _geometric_params()
            matrix = _affine_matrix(params, config.IMG_SIZE)
            landmarks = apply_geometry_to_landmarks(params, landmarks)
        else:
            params = matrix = None

        cache = self.frames
        frames = []
        for frame_id in frame_ids:
            img = np.array(cache[frame_id])   # copy out of the read-only map
            if self.augment:
                img = cv2.warpAffine(
                    img, matrix, (config.IMG_SIZE, config.IMG_SIZE),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )
                img = _photometric(img)
                if np.random.rand() < 0.25:
                    img = _random_erase(img)
            elif img.shape[0] != config.IMG_SIZE:
                img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                 interpolation=cv2.INTER_AREA)
            frames.append(self._to_chw(img))

        return (
            torch.from_numpy(np.stack(frames, axis=0)),          # (T, C, H, W) uint8
            torch.from_numpy(np.ascontiguousarray(landmarks)),   # (T, LANDMARK_DIM)
            label,
        )


def compute_class_weights(train_samples, num_classes):
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, _, label in train_samples:
        counts[label] += 1
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(train_samples, num_classes):
    """Per-sample weights for a balanced sampler (class I has the fewest frames)."""
    class_weights = compute_class_weights(train_samples, num_classes).numpy()
    return torch.tensor(
        [class_weights[label] for _, _, label in train_samples], dtype=torch.double
    )


if __name__ == "__main__":
    train_samples, val_samples, test_samples, label_map, class_counts, group_counts = build_splits()
    print(f"classes ({len(label_map)}):", list(label_map.values()))
    print("frames per class:", class_counts)
    print("capture sessions per class:", group_counts)
    print(f"windows -> train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
