"""Filename rules that decide what counts as a real sample and which
capture session it came from.

The raw dataset (Dataset/DATASET FELIX/) is a merge of several sources —
webcam bursts, phone photos, Roboflow exports — and someone also dropped
*offline-augmented copies* into the very same class folders:

    flip37.jpg / rotate37.jpg / augmented_image_37.jpg / A_5_aug3.jpg

Those are mirrored / rotated / recoloured versions of images that also live
in the folder as originals. Because the old split cut each class 70/15/15 by
sorted filename, the tail of that sort (dominated by `flip*` and `rotate*`)
became the test set — i.e. the model was scored on transformed copies of its
own training images. That is what produced the meaningless 96.99% figure.

This module centralises the two decisions that fix it, so preprocess.py,
dataset.py and audit_dataset.py can never disagree about them:

  is_derived()   -> drop offline-augmented copies entirely (augmentation is
                    done online during training instead, see dataset.py)
  source_group() -> which capture session a file belongs to, so a session is
                    split chronologically within itself and near-duplicate
                    frames never straddle a train/test boundary

Byte-identical duplicates (e.g. class L holds 172 exact copies named
`L_12(1).JPEG` alongside `L_12.JPEG`) are *not* handled here — filenames
cannot be trusted for that. preprocess.py hashes file contents instead.
"""
import re

import numpy as np

_NUM_RE = re.compile(r"(\d+)")


def natural_sort_key(name: str):
    """Sort `img2` before `img10` — keeps webcam bursts in capture order."""
    parts = _NUM_RE.split(name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# Offline-augmented copies, matched against the filename stem (no extension).
_DERIVED_RE = re.compile(
    r"""^(
          flip\d+                # flip37
        | rotate\d+              # rotate37
        | augmented_image_\d+    # augmented_image_37
        | .*_\d+_aug\d+          # A_5_aug3
        | .*\ copy               # image_7 copy
    )$""",
    re.IGNORECASE | re.VERBOSE,
)


def is_derived(stem: str) -> bool:
    """True for offline-augmented copies that must be excluded from the dataset."""
    return bool(_DERIVED_RE.match(stem))


# Roboflow / CVAT exports carry a content hash in the name, so every single
# file would otherwise become its own "group" and grouping would do nothing.
_RF_HASH_RE = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)
_UUIDISH_RE = re.compile(r"^.+\.[0-9a-f]+(-[0-9a-f]+){3,}$", re.IGNORECASE)


def source_group(stem: str) -> str:
    """Capture-session key for a filename stem.

    All files sharing a key are treated as one recording/export and are split
    chronologically *within* the group, so every visual condition (background,
    camera, lighting) is represented in train, val and test alike.
    """
    if _RF_HASH_RE.search(stem) or _UUIDISH_RE.match(stem):
        return "rf_export"
    key = _NUM_RE.sub("#", stem.lower())
    return re.sub(r"\s+", " ", key).strip()


# --------------------------------------------------------------------------
# perceptual near-duplicate detection
# --------------------------------------------------------------------------
#
# Filenames are not enough. The same photograph shows up under two different
# naming schemes (a Roboflow export re-exporting an image that also sits in
# the folder as `c_12.jpg`, for instance), so two copies of one picture can
# land in two different `source_group()` buckets and therefore in two
# different splits. Comparing pixels catches what names cannot.

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash_image(gray_small):
    """64-bit difference hash from a (size, size+1) grayscale image."""
    return np.packbits(gray_small[:, 1:] > gray_small[:, :-1])


def hamming_block(block, allh):
    """Pairwise bit distance between a block of hashes and every hash."""
    return _POPCOUNT[block[:, None, :] ^ allh[None, :, :]].sum(axis=2)


def cluster_near_duplicates(hashes, threshold, block=256):
    """Union-find clustering of frames whose dHash differs by <= threshold.

    Returns a list of clusters (index lists), only for clusters larger than a
    single frame — those are the ones that must not be split apart.
    """
    n = len(hashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for start in range(0, n, block):
        stop = min(start + block, n)
        dist = hamming_block(hashes[start:stop], hashes)
        for local_i, i in enumerate(range(start, stop)):
            for j in np.nonzero(dist[local_i] <= threshold)[0]:
                if j > i:
                    union(i, int(j))

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return [c for c in clusters.values() if len(c) > 1]
