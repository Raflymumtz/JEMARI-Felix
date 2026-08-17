// BISINDO real-time translator frontend.
// Captures webcam frames, streams them to the FastAPI backend over a
// WebSocket, and turns the stream of per-frame letter predictions into
// stable fingerspelled text using a client-side majority-vote debounce
// (mirrors config.VOTE_WINDOW / VOTE_MIN_AGREEMENT / CONF_THRESHOLD on
// the backend, but kept here so manual textarea edits are never clobbered
// by server-pushed state).

const CONFIG = {
  VOTE_WINDOW: 5,
  VOTE_MIN_AGREEMENT: 3,
  CONF_THRESHOLD: 0.55,
  NO_HAND_FRAMES_FOR_SPACE: 12,
  CAPTURE_INTERVAL_MS: 130,
  CAPTURE_WIDTH: 320,
  CAPTURE_HEIGHT: 240,
  JPEG_QUALITY: 0.7,
};

const els = {
  video: document.getElementById("video"),
  canvas: document.getElementById("canvas"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  handIndicator: document.getElementById("handIndicator"),
  letterOverlay: document.getElementById("letterOverlay"),
  confidenceFill: document.getElementById("confidenceFill"),
  confidenceValue: document.getElementById("confidenceValue"),
  latencyValue: document.getElementById("latencyValue"),
  outputText: document.getElementById("outputText"),
  speakBtn: document.getElementById("speakBtn"),
  spaceBtn: document.getElementById("spaceBtn"),
  backspaceBtn: document.getElementById("backspaceBtn"),
  clearBtn: document.getElementById("clearBtn"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
  alphabetSupport: document.getElementById("alphabetSupport"),
  aboutContent: document.getElementById("aboutContent"),
};

const ctx = els.canvas.getContext("2d");
els.canvas.width = CONFIG.CAPTURE_WIDTH;
els.canvas.height = CONFIG.CAPTURE_HEIGHT;

let ws = null;
let stream = null;
let captureTimer = null;
let latencyHistory = [];

// --- debounce state ---
let voteHistory = [];
let lastCommittedLetter = null;
let noHandCounter = 0;

function setStatus(state, text) {
  els.statusDot.classList.remove("ok", "bad");
  if (state) els.statusDot.classList.add(state);
  els.statusText.textContent = text;
}

function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/translate`);

  ws.onopen = () => setStatus("ok", "Terhubung ke server");
  ws.onclose = () => {
    setStatus("bad", "Terputus — mencoba lagi...");
    setTimeout(connectWebSocket, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (event) => handleServerMessage(JSON.parse(event.data));
}

function handleServerMessage(data) {
  if (data.type === "reset_ack") return;

  const { letter, confidence, hand_detected, client_ts } = data;

  if (typeof client_ts === "number") {
    const rtt = performance.now() - client_ts;
    latencyHistory.push(rtt);
    if (latencyHistory.length > 20) latencyHistory.shift();
    const avg = latencyHistory.reduce((a, b) => a + b, 0) / latencyHistory.length;
    els.latencyValue.textContent = `${avg.toFixed(0)} ms`;
  }

  els.handIndicator.textContent = hand_detected ? "Tangan terdeteksi" : "Tangan tidak terdeteksi";
  els.handIndicator.classList.toggle("active", !!hand_detected);

  els.letterOverlay.textContent = letter || "—";
  const pct = Math.round((confidence || 0) * 100);
  els.confidenceFill.style.width = `${pct}%`;
  els.confidenceValue.textContent = `${pct}%`;

  applyDebounce(letter, confidence, hand_detected);
}

function applyDebounce(letter, confidence, handDetected) {
  if (handDetected) {
    noHandCounter = 0;
  } else {
    noHandCounter += 1;
  }

  if (handDetected && letter && confidence >= CONFIG.CONF_THRESHOLD) {
    voteHistory.push(letter);
  } else {
    voteHistory.push(null);
  }
  if (voteHistory.length > CONFIG.VOTE_WINDOW) voteHistory.shift();

  // sustained absence of a hand => word boundary (insert a space once)
  if (noHandCounter === CONFIG.NO_HAND_FRAMES_FOR_SPACE) {
    const current = els.outputText.value;
    if (current && !current.endsWith(" ")) {
      els.outputText.value = current + " ";
    }
    lastCommittedLetter = null;
    voteHistory = [];
  }

  if (voteHistory.length === CONFIG.VOTE_WINDOW) {
    const counts = {};
    for (const v of voteHistory) {
      if (v === null) continue;
      counts[v] = (counts[v] || 0) + 1;
    }
    let candidate = null, best = 0;
    for (const [k, n] of Object.entries(counts)) {
      if (n > best) { best = n; candidate = k; }
    }
    if (candidate && best >= CONFIG.VOTE_MIN_AGREEMENT && candidate !== lastCommittedLetter) {
      els.outputText.value += candidate;
      lastCommittedLetter = candidate;
      voteHistory = [];
    }
  }
}

function captureAndSend() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (els.video.readyState < 2) return;

  ctx.drawImage(els.video, 0, 0, CONFIG.CAPTURE_WIDTH, CONFIG.CAPTURE_HEIGHT);
  const dataUrl = els.canvas.toDataURL("image/jpeg", CONFIG.JPEG_QUALITY);
  ws.send(JSON.stringify({ image: dataUrl, ts: performance.now() }));
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: CONFIG.CAPTURE_WIDTH, height: CONFIG.CAPTURE_HEIGHT, facingMode: "user" },
      audio: false,
    });
    els.video.srcObject = stream;
    els.startBtn.disabled = true;
    els.stopBtn.disabled = false;

    if (!ws || ws.readyState === WebSocket.CLOSED) connectWebSocket();
    captureTimer = setInterval(captureAndSend, CONFIG.CAPTURE_INTERVAL_MS);
  } catch (err) {
    alert("Tidak dapat mengakses kamera: " + err.message);
  }
}

function stopCamera() {
  if (captureTimer) clearInterval(captureTimer);
  captureTimer = null;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  els.letterOverlay.textContent = "—";
  els.handIndicator.textContent = "Tangan tidak terdeteksi";
  els.handIndicator.classList.remove("active");
}

function speakText() {
  const text = els.outputText.value.trim();
  if (!text) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "id-ID";
  speechSynthesis.cancel();
  speechSynthesis.speak(utterance);
}

async function loadClasses() {
  try {
    const res = await fetch("/api/classes");
    const data = await res.json();
    els.alphabetSupport.innerHTML = "";
    const allLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
    for (const l of allLetters) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = l;
      if (!data.classes.includes(l)) {
        chip.style.opacity = "0.3";
        chip.title = "Belum tersedia pada dataset saat ini";
      }
      els.alphabetSupport.appendChild(chip);
    }
  } catch (e) {
    console.warn("Gagal memuat daftar kelas", e);
  }
}

async function loadMetrics() {
  try {
    const res = await fetch("/api/metrics");
    const m = await res.json();
    if (!m || Object.keys(m).length === 0) {
      els.aboutContent.innerHTML = "<p>Model belum dilatih. Jalankan <code>train.py</code> di backend untuk menghasilkan metrik evaluasi.</p>";
      return;
    }
    els.aboutContent.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-card"><div class="label">Akurasi</div><div class="value">${(m.test_accuracy * 100).toFixed(1)}%</div></div>
        <div class="metric-card"><div class="label">Presisi (macro)</div><div class="value">${(m.test_precision_macro * 100).toFixed(1)}%</div></div>
        <div class="metric-card"><div class="label">Recall (macro)</div><div class="value">${(m.test_recall_macro * 100).toFixed(1)}%</div></div>
        <div class="metric-card"><div class="label">F1-score (macro)</div><div class="value">${(m.test_f1_macro * 100).toFixed(1)}%</div></div>
        <div class="metric-card"><div class="label">Latensi model</div><div class="value">${m.latency_ms_mean.toFixed(0)} ms</div></div>
        <div class="metric-card"><div class="label">Perangkat pelatihan</div><div class="value" style="font-size:0.95rem">${m.device}</div></div>
      </div>
      <p>Dievaluasi pada ${m.test_windows} sekuens uji dari ${m.num_classes} kelas huruf, jendela temporal ${m.window} frame @ ${m.img_size}x${m.img_size}px.</p>
    `;
  } catch (e) {
    els.aboutContent.innerHTML = "<p>Metrik belum tersedia.</p>";
  }
}

els.startBtn.addEventListener("click", startCamera);
els.stopBtn.addEventListener("click", stopCamera);
els.speakBtn.addEventListener("click", speakText);
els.spaceBtn.addEventListener("click", () => { els.outputText.value += " "; });
els.backspaceBtn.addEventListener("click", () => { els.outputText.value = els.outputText.value.slice(0, -1); });
els.clearBtn.addEventListener("click", () => {
  els.outputText.value = "";
  lastCommittedLetter = null;
  voteHistory = [];
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "reset" }));
});

connectWebSocket();
loadClasses();
loadMetrics();
