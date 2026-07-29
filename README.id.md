[English](README.md) · **Bahasa Indonesia** (versi English menyusul)

# Nyatet

Model span tagging hasil distilasi, cukup kecil untuk dikirim bersama aplikasi dan dijalankan di perangkat toko. Menggantikan API model besar untuk ekstraksi terstruktur dari teks informal Bahasa Indonesia.

> **Status:** implementasi belum dimulai. Dokumen ini adalah spesifikasi arsitektur dan hasil benchmark awal, bukan dokumentasi sistem yang sudah berjalan. Bagian yang belum dikerjakan didaftar di [Belum ada](#belum-ada).

![Model](https://img.shields.io/badge/model-IndoBERT--lite--p2-blue)
![Size](https://img.shields.io/badge/ONNX-42.6%20MB-green)
![Latency](https://img.shields.io/badge/latensi-18.3%20ms-green)
![Runtime](https://img.shields.io/badge/runtime-onnxruntime-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-pra--implementasi-orange)

---

## Masalah

Admin pemesanan di grosir sembako punya 40 sampai 80 pelanggan dagang tetap yang memesan ulang tiap minggu lewat pesan teks. Pukul 22:00 masuk enam puluh pesan, masing-masing ditulis dengan gaya yang sudah jadi kebiasaan pelanggan itu, lalu disalin tangan ke buku pesanan sebelum muat pukul 05:00.

Bentuk pesannya kira-kira begini:

```
bu indomi goreng 1 dus sm gula 5kg,
minyak 2 jrigen yg biasa
itu aja, besok pagi bisa?
```

Platform otomasi WhatsApp yang sudah ada (Dazo, WATI, Qiscus) menyelesaikan ini dengan cara membatasi input: tombol katalog, formulir, alur terpandu. Cara itu masuk akal untuk konsumen yang memang lagi belanja online. Tapi pelanggan dagang yang tiga tahun terakhir membeli dua belas barang yang sama dari supplier yang sama tidak akan mau mengklik katalog, dan tidak ada gunanya memaksa.

Jadi Nyatet mengambil jalan sebaliknya: biarkan mereka mengetik seperti biasa, lalu parsing pesannya.

---

## Rencana arsitektur

```
        pesan mentah
             │
             ▼
   ┌────────────────────────────┐
   │ tagger  (ONNX, 42.6 MB)    │   span BIO: ITEM / QTY / UNIT / ANAPHORIC
   └────────────────────────────┘
             │
             ▼
   ┌────────────────────────────┐
   │ normalizer                 │   "dua" -> 2, "seperempat" -> 0.25
   └────────────────────────────┘     (lookup table + regex, bukan model)
             │
             ▼
   ┌────────────────────────────┐
   │ resolver                   │   span -> SKU katalog + pengali kemasan
   └────────────────────────────┘
             │
             ▼
   order lines + tagging_confidence + resolution_confidence
```

Hanya tahap pertama yang menyentuh model. Dua tahap sisanya dirancang sebagai Python deterministik tanpa dependensi model, supaya sebagian besar pipeline bisa dites tanpa checkpoint apa pun.

---

## Keputusan desain

Bagian ini adalah keputusan yang sudah dikunci, dan jadi dasar implementasinya nanti.

### Kenapa tagging, bukan generate JSON

Model mengeluarkan span BIO di atas teks asli, bukan mengarang JSON.

- Setiap token keluaran menunjuk ke posisi di dalam input, jadi model tidak bisa berhalusinasi menyebut barang yang tidak ada di pesan.
- Ruang keluarannya terbatas, jadi encoder kecil sudah cukup. Tidak perlu API bermiliar parameter.
- Resolusi katalog jadi tahap terpisah yang bisa ditukar tanpa menyentuh model.

Pendekatan "suruh saja LLM keluarkan JSON" butuh model besar justru karena generation-nya tidak dibatasi. Begitu ruang keluaran dipersempit, model kecil jadi masuk akal.

### Empat tipe span

| Tipe | Contoh |
|---|---|
| `ITEM` | `indomi goreng` |
| `QTY` | `1`, `dua`, `seperempat` |
| `UNIT` | `dus`, `kg`, `renceng` |
| `ANAPHORIC` | `yg kyk kmrn`, `yg biasa`, `itu aja` |

### Aritmatika kuantitas tidak dipelajari model

`dua` jadi 2, `seperempat` / `¼` / `1/4` jadi 0.25. Itu tabel lookup dan regex, bukan sesuatu yang perlu dilatih.

Ukuran kemasan juga bukan pengetahuan bahasa, tapi **kolom katalog**. Satu `dus` mie instan dan satu `dus` sabun jelas beda isinya, dan yang tahu bedanya cuma katalog. Model men-tag span, normalizer mengonversi angka, katalog menyediakan pengalinya.

### Dua skor kepercayaan yang terpisah

Parsing dan resolusi bisa gagal sendiri-sendiri, jadi akan dilaporkan sendiri-sendiri juga.

| Skor | Sumber | Kegagalan yang ditangkap |
|---|---|---|
| Tagging confidence | Probabilitas span | Tidak sadar ada entitas di situ |
| Resolution confidence | Selisih top-1 vs top-2 | Sadar ada entitas, tapi tidak tahu SKU mana |

Kasus yang memotivasi pemisahan ini: token seperti `indomi` seharusnya ter-tag dengan yakin, karena posisinya sebagai nama barang jelas dari konteks kalimat. Tapi resolusinya justru ambigu, karena *Goreng*, *Soto*, dan *Ayam Bawang* sama-sama mirip. Kalau kedua hal itu digabung jadi satu angka, kasus ini akan terbaca seolah-olah parsing-nya jelek, padahal parsing-nya benar dan yang kurang cuma katalog.

---

## Hasil pengukuran

Validasi arsitektur pada IndoNLU NERP, dijalankan terpisah di Colab/Kaggle. Ini mengukur kelayakan pendekatan span tagging-nya, bukan task pemesanan. Kode training dan ekspor persis sama untuk kedua model, yang beda cuma nama checkpoint.

| | IndoBERT-lite-p2 (11,7 jt) | IndoBERT-base-p2 (124 jt) |
|---|---|---|
| F1, fp32 | 0.8040 | 0.8077 |
| F1, int8 (penuh) | 0.8014 | 0.8088 |
| Ukuran, fp32 | 42,6 MB | 472,7 MB |
| **Latensi, int8 (penuh)** | **18,3 ms** | **18,1 ms** |

Tiga hal yang keluar dari angka ini:

**Skala hampir tidak membantu di task ini.** Parameter 10,6 kali lebih banyak hanya menghasilkan 0.004 F1.

**Jumlah parameter menentukan ukuran file, bukan kecepatan.** Dua-duanya mendarat di ~18 ms setelah dikuantisasi. ALBERT memang memperkecil parameter dengan berbagi bobot antar-layer, tapi tetap menjalankan 12 layer. Seluruh percepatan datang dari kuantisasi, bukan dari pemilihan ukuran model.

**Kuantisasi tidak merusak akurasi.** Rentang lima varian ada di 0.8014 sampai 0.8106, masih di dalam variasi antar-run.

> **Catatan metrik.** Angka di atas adalah `seqeval` span-level strict. IndoNLU melaporkan task sequence labeling dengan metrik word-level, jadi angka ini **tidak bisa langsung dibandingkan** dengan tabel publikasi mereka tanpa penyelarasan metrik dulu.

### Catatan kuantisasi

`quantize_dynamic` default pada IndoBERT-lite cuma mengecilkan ukuran ~10%, jauh dari ekspektasi 4×, dan sempat bikin bingung cukup lama.

Setelah memeriksa tipe data initializer ONNX, penyebabnya ketemu: parameter sharing antar-layer di ALBERT membuat exporter mengeluarkan bobot transformer sebagai konstanta graph anonim (`onnx::MatMul_*`) yang dirujuk dua belas konsumen sekaligus. Quantizer dinamis bawaan ONNX Runtime melewatinya begitu saja, jadi yang terkuantisasi cuma tabel embedding.

Perbaikannya:

```python
quantize_dynamic(
    model_input, model_output,
    op_types_to_quantize=["MatMul"],
    extra_options={"MatMulConstBOnly": False},
)
```

Kuantisasi kembali penuh, tanpa biaya akurasi.

---

## Rencana struktur repo

```
training/
  order/            task target: generator data + fine-tune + ekspor
  benchmark/        validasi IndoNLU NERP
                    (dijalankan di mesin sendiri / Colab / Kaggle,
                     tidak pernah di dalam container yang di-serve)
serving/            FastAPI + inference, ONNX saja, tanpa torch
frontend/           satu halaman HTML, satu kotak teks, tanpa build step
tests/              pytest untuk seluruh komponen deterministik
docs/               PRD, ringkasan rulebook, catatan serah terima
docker-compose.yml  satu-satunya hal yang perlu dijalankan juri
```

`training/` dan `serving/` sengaja dipisah dependency tree-nya. `training/*/requirements.txt` isinya torch, transformers, datasets. `serving/requirements.txt` cuma `onnxruntime` + `transformers` (khusus tokenizer) + FastAPI. Image Docker tidak akan meng-install torch sama sekali, supaya start-up-nya cepat dan image-nya kecil.

Pemisahan AI / backend / frontend juga sengaja dibuat kebaca dari direktorinya: model cuma disentuh di satu file, dan semua tahap setelahnya deterministik.

---

## Risiko yang sudah diantisipasi

Ditulis di sini supaya tidak perlu ditanyakan:

- Pengelompokan baris pesanan direncanakan memakai heuristik posisional. Penempelan kuantitas pada pesan multi-item adalah masalah terbuka dan kemungkinan besar jadi sumber error utama.
- Konversi satuan hanya akan menangani satuan kemasan yang memang dideklarasikan katalog.
- Set evaluasi elisitasi kemungkinan tetap kecil (*n* ≈ 150) dan ditulis, bukan diambil dari operasional nyata.
- Evaluasi domain benchmark tidak menguji tahap normalizer maupun resolver, jadi angka di atas tidak mewakili akurasi end-to-end.
- Span bertipe event adalah kelas terlemah pada benchmark (EVT 0.388 F1).
- Angka latensi masih rata-rata dari 50 run. Perlu diukur ulang sebagai median dan p95.

---

## Belum ada

Urut kira-kira sesuai rencana pengerjaan:

- [ ] `training/order/generate_data.py`, generator data latih sintetik berbobot fenomena
- [ ] `training/order/train_tagger.py`, fine-tune tagger
- [ ] Set evaluasi elisitasi
- [ ] Model Order terlatih
- [ ] `export_onnx.py`, ekspor + kuantisasi ke `serving/models/tagger/`
- [ ] `normalizer.py` dan `resolver.py`, seluruh tahap deterministik
- [ ] Test suite untuk komponen deterministik
- [ ] Service FastAPI dan endpoint `/parse`
- [ ] `docker-compose.yml`
- [ ] Frontend satu halaman
- [ ] Tahap distilasi guru–murid (rencana awal fine-tune langsung dulu)
- [ ] Baseline pembanding RAG
- [ ] Tipe span `PACK_SIZE`, `PAYMENT_NOTE`, `LOCATION`, dijadwalkan sebagai peningkatan 10 jam di babak final

## Dokumen

| File | Isi |
|---|---|
| `docs/PRD.md` | Spesifikasi lengkap: arsitektur, rencana data, evaluasi, linimasa, risiko |
| `docs/rulebook-digest.md` | Aturan kompetisi, diekstrak dan disusun ulang |
| `docs/handoff.md` | Konteks perencanaan dan keputusan yang sudah dikunci |
