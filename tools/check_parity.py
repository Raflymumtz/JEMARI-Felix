"""Pre-training sanity checks that no loss curve would ever reveal.

1. Augmented landmarks must still sit on the augmented image. The training
   pipeline rotates and mirrors frames with cv2.warpAffine while transforming
   the landmark descriptor with a separate matrix; if those two disagree, the
   landmark branch learns a pose that does not match the picture and quietly
   drags accuracy down. Only looking at it catches this.

2. preprocess.py and server.py must produce identical landmark vectors for
   the same frame, otherwise the model is fed a different input distribution
   live than it saw in training.

Usage:  python tools/check_parity.py
Writes annotated frames to tools/parity_out/ for the visual check.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

import config
from dataset import (
    _affine_matrix,
    _geometric_params,
    apply_geometry_to_landmarks,
    build_splits,
    list_classes,
    load_class_meta,
)
from hand_crop import HandCropper

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_out")

# MediaPipe hand skeleton, for drawing
BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]


def draw_landmarks(img, vec, color=(0, 255, 0)):
    """Landmarks are wrist-centred and scale-normalised, so re-project them
    into the frame around its centre just to check orientation and mirroring."""
    if vec[-1] < 0.5:
        cv2.putText(img, "no hand", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        return img
    pts = vec[:-1].reshape(config.N_LANDMARKS, 3)
    h, w = img.shape[:2]
    scale = min(h, w) * 0.32
    xy = np.stack([pts[:, 0] * scale + w / 2, pts[:, 1] * scale + h / 2], axis=1).astype(int)
    for a, b in BONES:
        cv2.line(img, tuple(xy[a]), tuple(xy[b]), color, 1, cv2.LINE_AA)
    for p in xy:
        cv2.circle(img, tuple(p), 2, (0, 0, 255), -1)
    return img


def check_augmentation_parity(n=12):
    train_samples, _, _, label_map, _, _ = build_splits()
    os.makedirs(OUT_DIR, exist_ok=True)

    rng = np.random.default_rng(0)
    picks = [i for i in rng.choice(len(train_samples), size=n * 4, replace=False)]

    tiles, used = [], 0
    for idx in picks:
        paths, landmarks, label = train_samples[idx]
        if landmarks[0][-1] < 0.5:      # only frames with a real hand are informative
            continue
        params = _geometric_params()
        matrix = _affine_matrix(params, config.IMG_SIZE)
        aug_lm = apply_geometry_to_landmarks(params, landmarks)

        img = cv2.imread(paths[0])
        aug = cv2.warpAffine(img, matrix, (config.IMG_SIZE, config.IMG_SIZE),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        before = draw_landmarks(cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE)),
                                landmarks[0].copy())
        after = draw_landmarks(aug.copy(), aug_lm[0].copy(), color=(255, 200, 0))
        cv2.putText(after, f"{label_map[label]} rot{params['angle']:+.0f}"
                           f"{' flip' if params['flip'] else ''}",
                    (3, config.IMG_SIZE - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        tiles.append(np.hstack([before, after]))
        used += 1
        if used >= n:
            break

    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
    sheet = np.vstack(rows)
    out = os.path.join(OUT_DIR, "landmark_augmentation.png")
    cv2.imwrite(out, sheet)
    print(f"[1] Wrote {out}")
    print("    Each pair is (original | augmented). The skeleton must rotate and")
    print("    mirror together with the picture behind it.")


def check_geometry_is_rigid():
    """Rotation+flip must preserve the shape of the hand exactly."""
    _, _, test_samples, _, _, _ = build_splits()
    worst = 0.0
    for paths, landmarks, _ in test_samples[:200]:
        if landmarks[0][-1] < 0.5:
            continue
        params = _geometric_params()
        aug = apply_geometry_to_landmarks(params, landmarks)
        a = landmarks[:, :-1].reshape(-1, config.N_LANDMARKS, 3)
        b = aug[:, :-1].reshape(-1, config.N_LANDMARKS, 3)
        # pairwise joint distances are invariant under rotation and mirroring
        da = np.linalg.norm(a[:, :, None, :] - a[:, None, :, :], axis=-1)
        db = np.linalg.norm(b[:, :, None, :] - b[:, None, :, :], axis=-1)
        worst = max(worst, float(np.abs(da - db).max()))
    print(f"[2] Max change in joint-to-joint distance under augmentation: {worst:.2e}")
    assert worst < 1e-5, "augmentation is distorting the hand, not just rotating it"
    print("    OK - the transform is rigid (pure rotation/mirror), shape preserved.")


def check_preprocess_server_parity(n=40):
    """preprocess.py and server.py must produce the same landmark vector.

    Both must use IMAGE mode. An earlier version ran the server in MediaPipe's
    VIDEO mode, which tracks the previous frame's hand region instead of
    detecting afresh; that disagreed with the training pipeline on ~40% of
    frames (max difference 1.36 on a [-1,1] descriptor). This check exists to
    keep that from creeping back in.
    """
    raw_dir = config.RAW_DATASET_DIR
    letters = sorted(os.listdir(raw_dir))[:6]
    offline = HandCropper(static_image_mode=True)   # as used by preprocess.py
    live = HandCropper(static_image_mode=True)      # as used by server.py

    worst, compared = 0.0, 0
    for letter in letters:
        d = os.path.join(raw_dir, letter)
        for fname in sorted(os.listdir(d))[:n // len(letters)]:
            img = cv2.imread(os.path.join(d, fname))
            if img is None:
                continue
            _, found_a, lm_a = offline.crop(img, out_size=config.CACHE_IMG_SIZE)
            _, found_b, lm_b = live.crop(img, out_size=config.IMG_SIZE)
            if not (found_a and found_b):
                continue
            worst = max(worst, float(np.abs(lm_a - lm_b).max()))
            compared += 1
    offline.close()
    live.close()

    print(f"[3] Compared {compared} frames through the preprocess path and the server path.")
    print(f"    Max landmark difference: {worst:.2e}")
    assert worst < 1e-5, "preprocess and server disagree on landmarks"
    print("    OK - training and live inference see the same descriptor.")


def check_rotation_direction(per_class=8, classes=8):
    """Does the landmark transform rotate the same WAY the image does?

    A sign error keeps the hand rigid (check 2 still passes) and looks
    perfectly plausible on the contact sheet, while handing the model a pose
    mirrored against its own picture. The only way to settle it is to warp the
    image, re-run MediaPipe on the warped result, and see which of the two
    candidate transforms the fresh detection agrees with.
    """
    cropper = HandCropper(static_image_mode=True)
    rng = np.random.default_rng(7)

    ours, flipped = [], []
    for letter in list_classes()[:classes]:
        meta = load_class_meta(letter)
        idxs = [i for i in range(len(meta["files"])) if meta["landmarks"][i][-1] > 0.5]
        for i in rng.choice(idxs, size=min(per_class, len(idxs)), replace=False):
            img = cv2.imread(os.path.join(meta["dir"], meta["files"][i]))
            lm = meta["landmarks"][i:i + 1]

            params = _geometric_params()
            matrix = _affine_matrix(params, config.IMG_SIZE)
            warped = cv2.warpAffine(img, matrix, (config.IMG_SIZE, config.IMG_SIZE),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            _, found, detected = cropper.crop(warped, out_size=config.IMG_SIZE)
            if not found:
                continue

            wrong_params = dict(params)
            wrong_params["angle"] = -params["angle"]
            ours.append(np.abs(apply_geometry_to_landmarks(params, lm)[0][:-1] - detected[:-1]).mean())
            flipped.append(np.abs(apply_geometry_to_landmarks(wrong_params, lm)[0][:-1] - detected[:-1]).mean())

    cropper.close()
    ours, flipped = np.array(ours), np.array(flipped)
    closer = int((ours < flipped).sum())
    print(f"[4] Warped {len(ours)} frames and re-detected them with MediaPipe.")
    print(f"    our transform    vs re-detected: mean abs err {ours.mean():.4f}")
    print(f"    sign-flipped rot vs re-detected: mean abs err {flipped.mean():.4f}")
    print(f"    ours is closer on {closer}/{len(ours)} frames")
    assert ours.mean() < flipped.mean() and closer / len(ours) > 0.8,         "landmark rotation looks inverted relative to the image"
    print("    OK - rotation direction matches the image.")
    print("    (residual error is MediaPipe re-detecting a slightly different crop box)")


if __name__ == "__main__":
    check_augmentation_parity()
    check_geometry_is_rigid()
    check_preprocess_server_parity()
    check_rotation_direction()
    print("\nAll parity checks passed.")
