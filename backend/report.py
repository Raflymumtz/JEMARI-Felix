"""Full performance report for the trained BISINDO model.

Printed automatically at the end of train.py, and re-printable at any time:

    python report.py                 # everything
    python report.py --no-matrix     # skip the confusion matrix
    python report.py --csv           # also (re)write confusion_matrix.csv

Everything comes from saved_model/metrics.json, so the report always matches
the model currently being served.
"""
import argparse
import json
import os
import sys

import config

# The Windows console defaults to cp1252, which cannot encode box-drawing or
# bar characters. Everything printed here is therefore plain ASCII, and this
# guard makes sure an unexpected character degrades instead of raising.
try:
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass


def _pct(x):
    return f"{x * 100:.2f}%"


def _bar(value, width=18):
    """Small inline bar so weak letters stand out when skimming."""
    filled = int(round(value * width))
    return "#" * filled + "." * (width - filled)


def _rule(char="=", width=78):
    return char * width


def print_headline(m):
    print(_rule())
    print("HASIL EVALUASI PADA TEST SET")
    print(_rule())
    print(f"  Akurasi            {_pct(m['test_accuracy']):>9}")
    if "test_error_rate" in m:
        print(f"  Error rate         {_pct(m['test_error_rate']):>9}"
              f"   ({round(m['test_error_rate'] * m['test_windows'])} dari "
              f"{m['test_windows']} sekuens salah)")
    print()
    print(f"  {'':<20}{'macro':>10}{'weighted':>12}")
    for label, macro_key, weighted_key in (
        ("Presisi", "test_precision_macro", "test_precision_weighted"),
        ("Recall", "test_recall_macro", "test_recall_weighted"),
        ("F1-score", "test_f1_macro", "test_f1_weighted"),
    ):
        weighted = _pct(m[weighted_key]) if weighted_key in m else "-"
        print(f"  {label:<20}{_pct(m[macro_key]):>10}{weighted:>12}")
    print()
    print("  macro    = tiap huruf berbobot sama (adil untuk huruf berdata sedikit)")
    print("  weighted = berbobot jumlah sampel uji (mencerminkan test set apa adanya)")


def print_model_info(m):
    print()
    print(_rule())
    print("MODEL")
    print(_rule())
    print(f"  Arsitektur         CNN ({m.get('backbone', '?')}) + landmark + Transformer")
    print(f"  Parameter          {m.get('n_params', 0) / 1e6:.2f} juta")
    print(f"  Ukuran berkas      {m.get('model_size_mb', 0):.1f} MB")
    print(f"  Input              {m['window']} frame @ {m['img_size']}x{m['img_size']} px")
    print(f"  Kelas              {m['num_classes']} huruf")
    print(f"  Latensi inferensi  {m['latency_ms_mean']:.1f} ms/jendela "
          f"(p95 {m['latency_ms_p95']:.1f} ms) di {m['device']}")
    if "latency_ms_mean_cpu" in m:
        print(f"                     {m['latency_ms_mean_cpu']:.1f} ms/jendela di CPU")
    print(f"  Data               train={m['train_windows']} val={m['val_windows']} "
          f"test={m['test_windows']} sekuens")


def print_per_class(m):
    pc = m.get("per_class")
    if not pc:
        return
    print()
    print(_rule())
    print("PERFORMA PER HURUF")
    print(_rule())
    print(f"  {'Huruf':<7}{'Presisi':>9}{'Recall':>9}{'F1':>9}{'Uji':>6}   F1")
    for letter, v in sorted(pc.items()):
        print(f"  {letter:<7}{_pct(v['precision']):>9}{_pct(v['recall']):>9}"
              f"{_pct(v['f1']):>9}{v['support']:>6}   {_bar(v['f1'])}")

    ranked = sorted(pc.items(), key=lambda kv: kv[1]["f1"])
    weak = [f"{k} ({_pct(v['f1'])})" for k, v in ranked[:5]]
    print()
    print("  Terlemah: " + ", ".join(weak))
    supports = [v["support"] for v in pc.values()]
    print(f"  Catatan: tiap huruf hanya diuji dengan {min(supports)}-{max(supports)} sekuens, "
          f"jadi satu")
    print(f"  kesalahan menggeser F1 huruf itu sekitar "
          f"{100 / max(sum(supports) / len(supports), 1):.0f} poin. Baca tabel ini")
    print("  sebagai penunjuk huruf mana yang perlu tambahan data, bukan angka presisi.")


def print_confusion(m, cm=None, letters=None):
    # `cm` arrives as a numpy array from train.py and as a list of lists from
    # metrics.json, so length is the only safe emptiness test for both.
    cm = cm if cm is not None else m.get("confusion_matrix")
    letters = letters if letters is not None else m.get("classes")
    if cm is None or letters is None or len(cm) == 0 or len(letters) == 0:
        print("\n(Confusion matrix tidak tersedia - latih ulang untuk menghasilkannya.)")
        return
    cm = [list(row) for row in cm]
    letters = list(letters)

    print()
    print(_rule())
    print("CONFUSION MATRIX  (baris = huruf sebenarnya, kolom = prediksi model)")
    print(_rule())
    header = "      " + "".join(f"{c:>3}" for c in letters)
    print(header)
    for letter, row in zip(letters, cm):
        cells = "".join("  ." if v == 0 else f"{v:>3}" for v in row)
        total = sum(row)
        correct = row[letters.index(letter)]
        flag = "" if correct == total else "  <-"
        print(f"  {letter:<4}{cells}{flag}")
    print()
    print("  Titik (.) berarti nol. Angka di luar diagonal adalah kesalahan;")
    print("  baris bertanda <- adalah huruf yang belum sempurna.")

    mistakes = []
    for i, letter in enumerate(letters):
        for j, count in enumerate(cm[i]):
            if i != j and count:
                mistakes.append((count, letter, letters[j]))
    if mistakes:
        mistakes.sort(reverse=True)
        print()
        print("  Kesalahan paling sering:")
        for count, true_letter, pred_letter in mistakes[:8]:
            print(f"    {true_letter} dikira {pred_letter}  ({count}x)")


def print_hygiene(m):
    h = m.get("dataset_hygiene")
    if not h:
        return
    print()
    print(_rule())
    print("JEJAK AUDIT DATASET")
    print(_rule())
    print(f"  Berkas mentah                     {h['raw_files']:>7,}".replace(",", "."))
    print(f"  Augmentasi offline dibuang        {h['offline_augmented_excluded']:>7,}".replace(",", "."))
    print(f"  Duplikat identik dibuang          {h['exact_duplicates_excluded']:>7,}".replace(",", "."))
    print(f"  Gambar asli dipakai               {h['frames_used']:>7,}".replace(",", "."))
    print(f"  ... tangan terdeteksi MediaPipe   {h['frames_with_detected_hand']:>7,}".replace(",", "."))
    print()
    print("  Split dipotong per sesi pengambilan dan diverifikasi bebas gambar kembar")
    print("  antar-split (jalankan `python audit_dataset.py` untuk membuktikannya),")
    print("  sehingga angka di atas tidak dilebihkan oleh kebocoran data.")


def print_report(metrics, cm=None, letters=None, show_matrix=True):
    print()
    print_headline(metrics)
    print_model_info(metrics)
    print_per_class(metrics)
    if show_matrix:
        print_confusion(metrics, cm, letters)
    print_hygiene(metrics)
    print()


def main():
    parser = argparse.ArgumentParser(description="Print the full model performance report.")
    parser.add_argument("--no-matrix", action="store_true", help="skip the confusion matrix")
    parser.add_argument("--csv", action="store_true", help="also rewrite confusion_matrix.csv")
    args = parser.parse_args()

    if not os.path.exists(config.METRICS_PATH):
        raise SystemExit(f"{config.METRICS_PATH} not found - run `python train.py` first.")

    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)

    print_report(metrics, show_matrix=not args.no_matrix)

    if args.csv and metrics.get("confusion_matrix"):
        import csv
        with open(config.CONFUSION_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred"] + metrics["classes"])
            for letter, row in zip(metrics["classes"], metrics["confusion_matrix"]):
                writer.writerow([letter] + row)
        print(f"Confusion matrix ditulis ke {config.CONFUSION_PATH}")


if __name__ == "__main__":
    main()
