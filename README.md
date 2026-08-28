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
Dataset/DATASET FELIX/<HURUF>/*            (dataset mentah: 26 kelas, 32.906 berkas)
        |
        v  (backend/preprocess.py)
        |    - buang augmentasi offline & duplikat identik  -> 13.729 gambar asli
        |    - deteksi + crop tangan (MediaPipe Hand Landmarker)
        |    - simpan crop 160px + 21 landmark + kunci sesi + perceptual hash
        v
Dataset/PROCESSED/<HURUF>/{*.jpg, meta.npz}
        |
        v  (backend/audit_dataset.py — GERBANG: pastikan split tidak bocor)
        v  (backend/dataset.py   — split per sesi + augmentasi online)
        v  (backend/model.py     — MobileNetV3-Small + landmark + Transformer)
        v  (backend/train.py     — training GPU, metrik lengkap per kelas)
backend/saved_model/{model.pt, label_map.json, metrics.json, confusion_matrix.csv}
        |
        v  (backend/server.py — FastAPI + WebSocket, inferensi real-time)
frontend/  (webcam capture, tampilan huruf & teks, text-to-speech)
```

Model mengikuti arsitektur pada proposal (BAB 3.3): CNN mengekstraksi fitur
spasial per frame, lalu Transformer encoder (multi-head self-attention)
memodelkan hubungan temporal antar 8 frame berurutan sebelum diklasifikasikan
ke salah satu huruf BISINDO.

### Cabang landmark

Selain fitur CNN, setiap frame juga diwakili oleh **21 titik sendi tangan**
dari MediaPipe — koordinat yang memang sudah dihitung untuk menentukan kotak
crop, jadi gratis saat inferensi. Untuk ejaan jari, posisi sendi adalah
sinyal yang jauh lebih langsung daripada tekstur piksel, dan tetap andal saat
pencahayaan atau warna kulit menyimpang dari data latih.

```
frame ──→ MobileNetV3-Small ──→ pool ──→ Linear(576→128) ─┐
                                                          ├─→ Linear(192→192) ─┐
landmark(64) ──→ MLP(64→128→64) ──────────────────────────┘                    │
                                                                    8 token ───┤
                                                    PositionalEncoding + Transformer
                                                    (2 layer, 4 head, ff 384)  │
                                                                   mean-pool ──┤
                                                                    Linear(192→26)
```

Landmark dinormalisasi relatif kotak crop, digeser ke pergelangan, lalu
diskalakan — sehingga tidak bergantung pada posisi maupun jarak tangan ke
kamera. Frame yang tangannya gagal terdeteksi mendapat vektor nol dengan
**bit validitas 0**, sehingga model tahu cabang ini sedang buta dan bersandar
pada CNN saja (79,7% frame berhasil terdeteksi tangannya).

### Catatan penyimpangan dari proposal

1. **PyTorch, bukan TensorFlow.** TensorFlow tidak lagi mendukung GPU native
   di Windows sejak 2.10 (2022) dan versi itu tidak mengenali GPU RTX
   50-series (Blackwell, `sm_120`) di laptop ini. Agar "training menggunakan
   VGA" benar-benar berjalan di GPU, proyek ini memakai PyTorch build CUDA
   12.8.
2. **Cabang landmark ditambahkan** di samping CNN. Struktur CNN + Transformer
   pada proposal tetap utuh; ini penambahan satu jalur input, bukan
   penggantian.
3. **Backbone MobileNetV3-Small**, bukan MobileNetV2 — sekitar setengah
   parameter untuk akurasi setara, sehingga model lebih ringan dan lebih
   cepat dilayani ke banyak perangkat.

Metodologi, rancangan evaluasi, dan seluruh kerangka penelitian pada proposal
tidak berubah.

## Dataset dan kejujuran evaluasi

Ini bagian terpenting dari revisi ini dan layak dijelaskan di laporan akhir.

**Versi pertama proyek melaporkan akurasi 96,99%. Angka itu tidak sah.**
Folder dataset mentah ternyata mencampur **augmentasi offline** ke dalam
folder kelas yang sama dengan gambar aslinya:

| Pola nama berkas | Jumlah |
|---|---:|
| `<HURUF>_<n>_aug<m>.jpg` | 10.400 |
| `flip<n>.jpg` | 3.294 |
| `rotate<n>.jpg` | 3.095 |
| `augmented_image_<n>.jpg` | 2.321 |
| **Total augmentasi offline** | **18.719 (56,9%)** |

Karena versi lama memotong split 70/15/15 mengikuti urutan abjad nama berkas,
blok terakhir (test set) justru didominasi berkas `flip*` dan `rotate*` — yaitu
salinan cermin dan putar dari gambar yang ada di data latih. Model dinilai
memakai gambar yang sudah dihafalnya.

Selain itu ditemukan **458 duplikat byte-identical** (kelas L saja menyimpan
172 salinan persis bernama `L_12(1).JPEG` di samping `L_12.JPEG`).

### Yang dilakukan sekarang

1. **Augmentasi offline dibuang seluruhnya**; augmentasi dilakukan *online*
   saat training (crop-skala acak, rotasi ±20°, jitter warna, blur, noise
   kompresi JPEG, random erasing). Variasinya jauh lebih kaya dan secara
   struktural tidak mungkin bocor ke test set.
2. **Duplikat identik dibuang** lewat hash isi berkas.
3. **Split dipotong per sesi pengambilan**, bukan per kelas. Nama berkas
   dipakai untuk mengenali sesi (`0.jpg…108.jpg` burst webcam, `IMG_*.JPG`
   foto HP, `*.rf.<hash>.jpg` export Roboflow, `wall white (n)` sesi latar
   khusus). Sesi besar dipotong kronologis 70/15/15 dengan **embargo 8 frame**
   di setiap batas; sesi kecil dimasukkan utuh ke satu split saja.
4. **Verifikasi perceptual hash.** Nama berkas ternyata masih belum cukup —
   foto yang sama muncul di dua skema penamaan berbeda. Setiap gambar
   di-hash (dHash) dan setiap klaster kembar dipaksa berada di split yang
   sama. Tanpa langkah ini masih tersisa 125 pasangan bocor.

Sisa akhir: **13.729 gambar asli, 26 kelas (A–Z lengkap), 0 pasangan
near-duplicate yang menyeberangi batas split.**

Jalankan sendiri untuk memverifikasi:

```bash
python audit_dataset.py
```

Skrip ini menghitung ulang hash setiap frame langsung dari disk (tidak
mempercayai cache yang dipakai pembagi split) dan **keluar dengan status
gagal** bila menemukan satu saja gambar kembar antar-split. Angka akurasi di
bawah baru bermakna karena gerbang ini lolos.

## Hasil

<!-- RESULTS -->

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

1. **Preprocessing** (deteksi & crop tangan + landmark; ±3 menit untuk 26
   kelas). Sudah di-cache di `Dataset/PROCESSED/`, jalankan ulang hanya jika
   dataset mentah berubah:
   ```bash
   python preprocess.py              # hanya kelas yang berubah
   python preprocess.py --force      # ulangi semuanya
   python preprocess.py --only L,N,P # kelas tertentu saja
   ```

2. **Audit** — gerbang wajib sebelum training:
   ```bash
   python audit_dataset.py
   ```

3. **Training** (pakai GPU otomatis jika tersedia; ±20 menit untuk 40 epoch):
   ```bash
   python train.py
   ```
   Menghasilkan `saved_model/model.pt`, `label_map.json`, `metrics.json`,
   `history.csv`, dan `confusion_matrix.csv`.

4. **Jalankan website**:
   ```bash
   python server.py
   ```
   Buka `http://localhost:8000` (izinkan akses kamera).

### Uji integritas (opsional tapi disarankan setelah mengubah augmentasi)

```bash
python ../tools/check_parity.py
```

Memeriksa empat hal yang tidak akan pernah terlihat dari kurva loss:
augmentasi landmark benar-benar rigid, arahnya sama dengan rotasi gambar,
preprocessing dan server menghasilkan deskriptor identik, dan hasilnya bisa
diperiksa mata lewat `tools/parity_out/landmark_augmentation.png`.

## Membuka dari HP (Chrome Android)

**Ini sering gagal dan penyebabnya bukan kode aplikasi.** Chrome hanya
mengizinkan akses kamera pada *secure context*: `localhost` atau HTTPS.
Membuka `http://192.168.x.x:8000` dari HP akan **selalu ditolak izin
kameranya**, seberapa pun benar kodenya.

**Cara yang disarankan** — terowongan HTTPS tepercaya, tanpa akun:

```bash
python server.py                              # terminal 1
cloudflared tunnel --url http://localhost:8000 # terminal 2
```

Buka URL `https://….trycloudflare.com` yang muncul di HP.

**Cadangan tanpa internet** — sertifikat self-signed di jaringan lokal:

```bash
python server.py --https
```

Lalu buka `https://<IP-laptop>:8000` di HP dan lewati peringatan sertifikat
(Advanced → Proceed). Kurang mulus, tapi berfungsi saat tidak ada internet.

## Cara kerja real-time di website

- Browser menangkap frame webcam (~7–8 fps) dan mengirimkannya lewat
  WebSocket (`/ws/translate`) ke server. Laju kirim **menyesuaikan diri**
  dengan latensi terukur, sehingga koneksi HP yang lambat tidak menumpuk
  antrean frame.
- Server mendeteksi & crop tangan (MediaPipe, mode IMAGE — sama persis
  dengan yang dipakai saat preprocessing), menjalankan CNN + cabang landmark
  untuk setiap frame, lalu Transformer atas jendela 8 frame terakhir.
  Inferensi berjalan di thread terpisah agar satu klien lambat tidak menahan
  klien lain, dan frame yang sudah basi dibuang alih-alih diproses.
- Prediksi dikirim balik bersama timestamp pengiriman asli, sehingga latensi
  yang ditampilkan adalah latensi end-to-end sungguhan (akuisisi citra →
  hasil ditampilkan), sesuai definisi pada BAB 3.4 proposal.
- Browser melakukan voting mayoritas atas beberapa prediksi terakhir sebelum
  "mengunci" sebuah huruf ke kotak teks, dan otomatis menyisipkan spasi saat
  tangan menjauh cukup lama.
- Tombol **Ucapkan** membacakan hasil memakai Web Speech API (`lang="id-ID"`).
- Panel **"Tentang Penelitian & Performa Model"** menampilkan metrik
  keseluruhan, tabel per huruf, dan ringkasan kebersihan data langsung dari
  `metrics.json`.

## Keterbatasan yang jujur perlu diketahui

- Dataset berisi gambar statis per huruf (fingerspelling alfabet), bukan
  video kalimat/kata BISINDO utuh — jadi sistem ini adalah pengenal **ejaan
  huruf** real-time, bukan penerjemah kalimat BISINDO penuh. Modul
  Transformer tetap dipakai secara nyata: ia menstabilkan prediksi antar-frame
  dari aliran webcam, bukan sekadar mengklasifikasi satu foto.
- **Test set kecil.** Setelah semua penyaringan, hanya tersisa sekitar 9
  sekuens uji per huruf. Artinya satu sekuens salah menggeser F1 huruf itu
  sekitar 10 poin — angka per-kelas harus dibaca sebagai indikasi kasar,
  bukan pengukuran presisi. Menambah data uji adalah perbaikan paling
  berharga berikutnya.
- **Kelas I paling sedikit datanya** (256 gambar asli, versus 460–640 untuk
  kelas lain) dan paling layak ditambah.
- Ditemukan 7 pasang gambar dari huruf **berbeda** yang nyaris identik secara
  visual (label noise). Jumlahnya kecil, tetapi menandakan ada gambar
  ambigu/salah label di dataset sumber.
- 20,3% frame tidak terdeteksi tangannya oleh MediaPipe dan memakai fallback
  center-crop dengan cabang landmark dinonaktifkan.
- Server memproses satu koneksi WebSocket per klien di thread pool — cukup
  untuk demo/skala penelitian, belum dioptimalkan untuk banyak pengguna
  bersamaan.
