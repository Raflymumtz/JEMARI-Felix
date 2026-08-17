"""Shared hand-detection + cropping logic.

Used by both offline preprocessing (preprocess.py) and the real-time
inference server (server.py) so a frame is turned into a model-ready crop
in exactly the same way in both places.

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

    def crop(self, img_bgr: np.ndarray):
        """Returns (crop_bgr resized to IMG_SIZE, hand_found: bool)."""
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
        else:
            # fallback: center square crop of the full frame
            side = min(h, w)
            cx0, cy0 = (w - side) // 2, (h - side) // 2
            crop = img_bgr[cy0:cy0 + side, cx0:cx0 + side]
            hand_found = False

        crop = cv2.resize(crop, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
        return crop, hand_found

    def close(self):
        self._landmarker.close()
