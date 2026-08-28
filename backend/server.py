"""Real-time BISINDO translation server.

Serves the frontend (frontend/) and exposes:
  GET  /api/classes   -> list of supported letters
  GET  /api/metrics   -> training metrics (accuracy/precision/recall/F1/latency)
  GET  /api/status    -> model / device information
  WS   /ws/translate  -> streams webcam frames from the browser, returns the
                         live-predicted letter, the accumulated fingerspelled
                         text, and end-to-end latency (echoes the client's
                         send timestamp so latency = capture -> result shown,
                         exactly the definition used in the proposal's
                         evaluation section).

Run:
    python server.py                 # http://localhost:8000
    python server.py --https         # self-signed TLS, for LAN testing

Opening the page from a phone needs a *secure context*, otherwise Chrome
refuses camera access no matter what the page does. `http://localhost` counts
as secure; `http://192.168.x.x` does not. See the README for the recommended
tunnel-based setup.
"""
import argparse
import asyncio
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
from dataset import normalize_frames
from hand_crop import HandCropper
from model import SignCNNTransformer

FRONTEND_DIR = os.path.join(config.BASE_DIR, "frontend")

app = FastAPI(title="Penerjemah Abjad BISINDO Real-Time")
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

        # A checkpoint and a label map from different runs load without error
        # and then mislabel every prediction, which is far worse than not
        # serving at all — refuse the pair instead.
        if checkpoint["num_classes"] != len(LABEL_MAP):
            MODEL = None
            print(f"ERROR: {config.MODEL_PATH} has {checkpoint['num_classes']} classes but "
                  f"{config.LABEL_MAP_PATH} lists {len(LABEL_MAP)}. They are from different "
                  f"training runs; predictions would be mislabelled. Re-run train.py.")
            return

        model = SignCNNTransformer(
            num_classes=checkpoint["num_classes"],
            backbone=checkpoint.get("backbone"),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(DEVICE)
        model.eval()
        MODEL = model
        print(f"Loaded trained model from {config.MODEL_PATH} on {DEVICE} "
              f"({checkpoint['num_classes']} classes, {model.backbone_name})")
    else:
        MODEL = None
        print("WARNING: no trained model found yet — run train.py first. "
              "The server will still start but predictions will be disabled.")


load_model()


def frame_to_tensor(crop_bgr):
    """Same normalisation the training pipeline applies, via the same helper."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    chw = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))))
    return normalize_frames(chw.unsqueeze(0).to(DEVICE))[0]


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
        "backbone": MODEL.backbone_name if MODEL is not None else None,
    }


def infer(frame, cropper, embedding_buffer):
    """Crop, embed and (once the window is full) classify a frame.

    Runs in a worker thread: the torch call is synchronous and would otherwise
    block the event loop, so one slow phone would stall every other client.
    """
    crop, hand_found, landmarks = cropper.crop(frame, out_size=config.IMG_SIZE)

    letter, confidence = None, 0.0
    if MODEL is not None:
        with torch.no_grad():
            tensor = frame_to_tensor(crop)
            lm_tensor = torch.from_numpy(landmarks).float().to(DEVICE)
            embedding_buffer.append(MODEL.embed_frame(tensor, lm_tensor))

            if len(embedding_buffer) == config.WINDOW:
                stacked = torch.stack(list(embedding_buffer), dim=0)
                probs = torch.softmax(MODEL.classify_embeddings(stacked), dim=0)
                top_idx = int(torch.argmax(probs).item())
                confidence = float(probs[top_idx].item())
                letter = LABEL_MAP.get(top_idx)

    return letter, confidence, hand_found


@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    await ws.accept()

    # IMAGE mode, not VIDEO mode. The training set is a pile of unrelated
    # stills, so preprocess.py ran fresh detection on every one of them.
    # MediaPipe's VIDEO mode reuses the previous frame's hand region, which
    # produced a visibly different landmark vector on ~40% of frames in a
    # parity test — the model would be fed a descriptor it never saw during
    # training. Fresh detection costs ~4 ms more per frame and removes that
    # skew entirely (see tools/check_parity.py).
    cropper = HandCropper(static_image_mode=True)

    embedding_buffer = deque(maxlen=config.WINDOW)
    # Debounce/word-segmentation (majority-vote over recent predictions,
    # space-on-no-hand) is handled client-side in app.js so that manual
    # edits to the output textarea are never clobbered by server state.
    # The server only owns what genuinely requires the model: the rolling
    # window of frame embeddings feeding the Transformer.

    state = {"latest": None, "closed": False}
    has_frame = asyncio.Event()

    async def reader():
        """Receive frames, keeping only the newest one.

        On a phone over wifi the browser can queue frames faster than the model
        drains them. Without dropping the backlog, displayed latency climbs and
        never recovers, and the letter shown lags behind the hand on screen.
        """
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                if msg.get("type") == "reset":
                    embedding_buffer.clear()
                    state["latest"] = None
                    await ws.send_json({"type": "reset_ack"})
                    continue
                state["latest"] = msg      # overwrite: stale frames are dropped
                has_frame.set()
        except Exception:
            state["closed"] = True
            has_frame.set()

    reader_task = asyncio.create_task(reader())

    try:
        while True:
            await has_frame.wait()
            has_frame.clear()
            if state["closed"]:
                break

            msg, state["latest"] = state["latest"], None
            if msg is None:
                continue

            frame = decode_data_url(msg["image"])
            letter, confidence, hand_found = await asyncio.to_thread(
                infer, frame, cropper, embedding_buffer
            )

            await ws.send_json({
                "letter": letter,
                "confidence": confidence,
                "hand_detected": hand_found,
                "client_ts": msg.get("ts"),
                "server_ts": time.time() * 1000.0,
            })
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        reader_task.cancel()
        cropper.close()


# static frontend last, so the API/WS routes above take priority
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def ensure_self_signed_cert(cert_dir):
    """Create a throwaway certificate so phones on the LAN get a secure context."""
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "dev-cert.pem")
    key_path = os.path.join(cert_dir, "dev-key.pem")
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "bisindo-dev")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    print(f"Generated self-signed certificate in {cert_dir}")
    return cert_path, key_path


def main():
    parser = argparse.ArgumentParser(description="Run the BISINDO translation server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--https", action="store_true",
                        help="serve over TLS with a self-signed certificate")
    args = parser.parse_args()

    import uvicorn

    ssl_kwargs = {}
    if args.https:
        cert, key = ensure_self_signed_cert(os.path.join(os.path.dirname(__file__), "certs"))
        ssl_kwargs = {"ssl_certfile": cert, "ssl_keyfile": key}
        print("\nServing over HTTPS with a self-signed certificate.")
        print("Chrome will show a warning — tap Advanced -> Proceed to accept it.")
        print("For a demo, a real certificate via `cloudflared tunnel --url "
              "http://localhost:8000` is more reliable (see README).\n")

    uvicorn.run(app, host=args.host, port=args.port, **ssl_kwargs)


if __name__ == "__main__":
    main()
