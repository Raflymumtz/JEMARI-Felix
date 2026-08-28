"""Shared hand-detection + cropping logic.

Used by both offline preprocessing (preprocess.py) and the real-time
inference server (server.py) so a frame is turned into a model-ready crop
in exactly the same way in both places.

Besides the crop, this also returns a normalised **landmark vector**: the 21
hand joints MediaPipe already computes to find the crop box, reshaped into a
translation- and scale-invariant descriptor. Those joint positions are the
single most discriminative signal for fingerspelling, and they cost nothing
extra at inference because the detector has to run anyway.

Uses the MediaPipe Tasks API (mediapipe>=0.10 removed the old
`mp.solutions.hands` API in favor of `HandLandmarker`).
"""
import os
import time

import cv2
import mediapipe as mp
import numpy as np

import config

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_models", "hand_landmarker.task")

# Landmark vector when no hand was found: all-zero coordinates plus a 0
# validity flag, so the model can tell "hand at the origin" apart from
# "landmark branch is blind, rely on the CNN".
BLIND_LANDMARKS = np.zeros(config.LANDMARK_DIM, dtype=np.float32)


def normalize_landmarks(landmarks, img_w, img_h, x0, x1, y0, y1):
    """Turn MediaPipe landmarks into a crop-relative, scale-free descriptor.

    Both preprocess.py and server.py must produce *identical* vectors for the
    same hand, otherwise the model silently sees a different input distribution
    at inference than it was trained on. Keeping this in one function is what
    guarantees that (see the parity check in tools/check_parity.py).
    """
    box_w = max(x1 - x0, 1)
    box_h = max(y1 - y0, 1)

    pts = np.empty((config.N_LANDMARKS, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        # image-normalised -> pixels -> crop-box-relative
        pts[i, 0] = (lm.x * img_w - x0) / box_w
        pts[i, 1] = (lm.y * img_h - y0) / box_h
        # MediaPipe's z is scaled like x (relative to image width)
        pts[i, 2] = (lm.z * img_w) / box_w

    # wrist (joint 0) becomes the origin -> translation invariant
    pts -= pts[0]

    # divide by the furthest joint -> invariant to hand size / distance
    scale = float(np.max(np.linalg.norm(pts, axis=1)))
    if scale > 1e-6:
        pts /= scale

    vec = np.empty(config.LANDMARK_DIM, dtype=np.float32)
    vec[:-1] = pts.reshape(-1)
    vec[-1] = 1.0  # validity flag
    return vec


class HandCropper:
    def __init__(self, static_image_mode=False):
        vision = mp.tasks.vision
        base_options = mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=1,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._static = static_image_mode
        self._last_ts_ms = -1

    def crop(self, img_bgr: np.ndarray, out_size: int = None):
        """Returns (crop_bgr resized to out_size, hand_found, landmark vector).

        `out_size` defaults to config.CACHE_IMG_SIZE for offline preprocessing;
        the server passes config.IMG_SIZE to feed the model directly.
        """
        if out_size is None:
            out_size = config.CACHE_IMG_SIZE

        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._static:
            result = self._landmarker.detect(mp_image)
        else:
            ts_ms = int(time.time() * 1000)
            if ts_ms <= self._last_ts_ms:
                ts_ms = self._last_ts_ms + 1
            self._last_ts_ms = ts_ms
            result = self._landmarker.detect_for_video(mp_image, ts_ms)

        landmark_vec = BLIND_LANDMARKS

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            box_w = x_max - x_min
            box_h = y_max - y_min
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            side = max(box_w, box_h) * (1 + config.HAND_MARGIN)

            x0 = int(np.clip((cx - side / 2) * w, 0, w - 1))
            x1 = int(np.clip((cx + side / 2) * w, 1, w))
            y0 = int(np.clip((cy - side / 2) * h, 0, h - 1))
            y1 = int(np.clip((cy + side / 2) * h, 1, h))

            if x1 - x0 < 4 or y1 - y0 < 4:
                crop = img_bgr
                hand_found = False
            else:
                crop = img_bgr[y0:y1, x0:x1]
                hand_found = True
                landmark_vec = normalize_landmarks(landmarks, w, h, x0, x1, y0, y1)
        else:
            # fallback: center square crop of the full frame
            side = min(h, w)
            cx0, cy0 = (w - side) // 2, (h - side) // 2
            crop = img_bgr[cy0:cy0 + side, cx0:cx0 + side]
            hand_found = False

        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return crop, hand_found, landmark_vec

    def close(self):
        self._landmarker.close()
