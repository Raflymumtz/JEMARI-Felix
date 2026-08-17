# Penerjemah BISINDO Real-Time

Implementasi kerja dari proposal penelitian OPSI 2026:

> **Pengembangan Model Translasi Bahasa Isyarat Indonesia Berbasis Computer
> Vision dan Transformer Architecture sebagai Sistem Penerjemah Real-Time
> untuk Aksesibilitas Komunikasi Digital**
> Felix Aria & Muhammad Raysar Al-Fatih — SMAN 1 Balikpapan

Proyek ini berisi pipeline lengkap: preprocessing dataset, pelatihan model
CNN + Transformer dengan akselerasi GPU, dan aplikasi web real-time yang
menerjemahkan ejaan huruf BISINDO dari webcam menjadi teks + suara.

## Arsitektur

```
Dataset/DATASET FELIX/<HURUF>/*.jpg   (dataset asli: 25 kelas huruf, ~29.500 gambar)
        |
        v  (backend/preprocess.py — deteksi & crop tangan dengan MediaPipe)
Dataset/PROCESSED/<HURUF>/*.jpg
        |
        v  (backend/dataset.py — sliding window 8 frame, split 70/15/15 per kelas)
        v  (backend/model.py   — MobileNetV2 CNN + Transformer self-attention)
        v  (backend/train.py   — training GPU, metrik accuracy/precision/recall/F1/latensi)
backend/saved_model/{model.pt, label_map.json, metrics.json}
        |
        v  (backend/server.py — FastAPI + WebSocket, inferensi real-time)
frontend/  (webcam capture, tampilan huruf & teks, text-to-speech)
```

Model mengikuti persis arsitektur pada proposal (BAB 3.3): MobileNetV2
mengekstraksi fitur spasial per frame, lalu sebuah Transformer encoder
(multi-head self-attention) memodelkan hubungan temporal antar 8 frame
berurutan sebelum diklasifikasikan ke salah satu huruf BISINDO.

**Catatan penyimpangan dari proposal:** proposal menyebut TensorFlow sebagai
framework, namun TensorFlow sudah tidak mendukung GPU native di Windows
sejak versi 2.10 (2022) dan versi itu tidak mengenali GPU RTX 50-series
(arsitektur Blackwell, `sm_120`) di laptop ini. Agar "training menggunakan
VGA" benar-benar berjalan di GPU, proyek ini memakai **PyTorch** dengan
build CUDA 12.8, yang punya dukungan native Windows + Blackwell. Arsitektur
CNN+Transformer, metodologi, dan seluruh rancangan pada proposal tidak
berubah — hanya framework implementasinya.

## Dataset

Dataset asli (`Dataset/DATASET FELIX/`) berisi 25 kelas huruf (A–Z tanpa
huruf **N**, yang belum ada datanya) dengan total ±29.500 frame gambar
tangan. Sebanyak 78,8% frame berhasil dideteksi tangannya oleh MediaPipe
Hand Landmarker (sisanya memakai fallback center-crop). Lihat
`Dataset/PROCESSED/manifest.json` untuk rincian per kelas.

## Setup

```bash
cd backend
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verifikasi GPU terdeteksi:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Menjalankan pipeline

1. **Preprocessing** (deteksi & crop tangan — sudah dijalankan sekali, cache
   di `Dataset/PROCESSED/`; jalankan ulang hanya jika dataset mentah berubah):
   ```bash
   python preprocess.py
   ```

2. **Training** (pakai GPU otomatis jika tersedia):
   ```bash
   python train.py
   ```
   Menghasilkan `backend/saved_model/model.pt`, `label_map.json`,
   `metrics.json`, dan `history.csv` (kurva training per epoch).

3. **Jalankan website**:
   ```bash
   python server.py
   ```
   Buka `http://localhost:8000` di browser (izinkan akses kamera).

## Cara kerja real-time di website

- Browser menangkap frame webcam (~7–8 fps) dan mengirimkannya lewat
  WebSocket (`/ws/translate`) ke server.
- Server mendeteksi & crop tangan (MediaPipe), lalu menjalankan CNN untuk
  setiap frame dan Transformer atas jendela 8 frame terakhir untuk
  memprediksi huruf + tingkat keyakinan.
- Prediksi dikirim balik ke browser bersama timestamp pengiriman asli,
  sehingga latensi yang ditampilkan adalah latensi end-to-end sungguhan
  (akuisisi citra → hasil ditampilkan), sesuai definisi pada BAB 3.4 proposal.
- Browser melakukan voting mayoritas atas beberapa prediksi terakhir
  sebelum "mengunci" sebuah huruf ke kotak teks (mengurangi flicker), dan
  otomatis menyisipkan spasi saat tangan menjauh dari kamera cukup lama.
- Tombol **Ucapkan** membacakan teks hasil terjemahan memakai Web Speech
  API bawaan browser (`lang="id-ID"`).
- Bagian **"Tentang Penelitian & Performa Model"** di halaman menampilkan
  akurasi/presisi/recall/F1/latensi hasil evaluasi `train.py` secara
  langsung dari `metrics.json`.

## Keterbatasan yang jujur perlu diketahui

- Dataset berisi gambar statis per huruf (fingerspelling alfabet), bukan
  video kalimat/kata BISINDO utuh — jadi sistem ini adalah pengenal
  **ejaan huruf** real-time, bukan penerjemah kalimat BISINDO penuh.
  Modul Transformer tetap dipakai secara nyata: ia menstabilkan prediksi
  antar-frame dari aliran webcam, bukan sekadar mengklasifikasi satu foto.
- Huruf **N** belum didukung karena tidak ada datanya (ditandai pudar di
  UI). Beberapa huruf (L, P) punya jumlah sampel jauh lebih sedikit
  (170 dan 263 frame) sehingga akurasinya kemungkinan lebih rendah —
  ini tercermin di metrik per-kelas dan pantas disebut sebagai batasan
  penelitian pada laporan akhir.
- Server saat ini memproses satu koneksi WebSocket secara sinkron per
  frame — cukup untuk demo/skala penelitian, belum dioptimalkan untuk
  banyak pengguna bersamaan.
