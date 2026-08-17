"""Real-time BISINDO translation server.

Serves the frontend (frontend/) and exposes:
  GET  /api/classes   -> list of supported letters
  GET  /api/metrics   -> training metrics (accuracy/precision/recall/F1/latency)
  WS   /ws/translate  -> streams webcam frames from the browser, returns the
                         live-predicted letter, the accumulated fingerspelled
                         text, and end-to-end latency (echoes the client's
                         send timestamp so latency = capture -> result shown,
                         exactly the definition used in the proposal's
                         evaluation section).

Run:
    python server.py
Then open http://localhost:8000 in a browser (camera permission required).
"""
import base64
import json
import os
import time
from collections import deque

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import config
from hand_crop import HandCropper
from model import SignCNNTransformer

FRONTEND_DIR = os.path.join(config.BASE_DIR, "frontend")

app = FastAPI(title="BISINDO Real-Time Translator")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
LABEL_MAP = {}
METRICS = {}


def load_model():
    global MODEL, LABEL_MAP, METRICS

    if os.path.exists(config.LABEL_MAP_PATH):
        with open(config.LABEL_MAP_PATH) as f:
            LABEL_MAP = {int(k): v for k, v in json.load(f).items()}
    else:
        LABEL_MAP = {}

    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH) as f:
            METRICS = json.load(f)

    if os.path.exists(config.MODEL_PATH) and LABEL_MAP:
        checkpoint = torch.load(config.MODEL_PATH, map_location=DEVICE)
        model = SignCNNTransformer(num_classes=checkpoint["num_classes"], pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(DEVICE)
        model.eval()
        MODEL = model
        print(f"Loaded trained model from {config.MODEL_PATH} on {DEVICE}")
    else:
        MODEL = None
        print("WARNING: no trained model found yet — run train.py first. "
              "The server will still start but predictions will be disabled.")


load_model()


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def frame_to_tensor(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).float().to(DEVICE)


def decode_data_url(data_url: str):
    header, b64data = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64data)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.get("/api/classes")
def get_classes():
    return {"classes": [LABEL_MAP[i] for i in sorted(LABEL_MAP)]}


@app.get("/api/metrics")
def get_metrics():
    return METRICS


@app.get("/api/status")
def get_status():
    return {
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_classes": len(LABEL_MAP),
    }


@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    await ws.accept()
    cropper = HandCropper(static_image_mode=False)

    embedding_buffer = deque(maxlen=config.WINDOW)
    # Debounce/word-segmentation (majority-vote over recent predictions,
    # space-on-no-hand) is handled client-side in app.js so that manual
    # edits to the output textarea are never clobbered by server state.
    # The server only owns what genuinely requires the model: the rolling
    # window of CNN embeddings feeding the Transformer.

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "reset":
                embedding_buffer.clear()
                await ws.send_json({"type": "reset_ack"})
                continue

            client_ts = msg.get("ts")
            frame = decode_data_url(msg["image"])

            crop, hand_found = cropper.crop(frame)
            letter, confidence = None, 0.0

            if MODEL is not None:
                with torch.no_grad():
                    tensor = frame_to_tensor(crop)
                    emb = MODEL.embed_frame(tensor)
                    embedding_buffer.append(emb)

                    if len(embedding_buffer) == config.WINDOW:
                        stacked = torch.stack(list(embedding_buffer), dim=0)
                        logits = MODEL.classify_embeddings(stacked)
                        probs = torch.softmax(logits, dim=0)
                        top_idx = int(torch.argmax(probs).item())
                        confidence = float(probs[top_idx].item())
                        letter = LABEL_MAP.get(top_idx)

            await ws.send_json({
                "letter": letter,
                "confidence": confidence,
                "hand_detected": hand_found,
                "client_ts": client_ts,
                "server_ts": time.time() * 1000.0,
            })
    except WebSocketDisconnect:
        pass
    finally:
        cropper.close()


# static frontend last, so the API/WS routes above take priority
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
