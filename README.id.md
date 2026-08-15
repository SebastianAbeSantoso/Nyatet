[English](README.md) · **Bahasa Indonesia**

# Nyatet

Model span tagging 11 MB yang berjalan offline di perangkat penjual, untuk
mengubah pesan pemesanan WhatsApp berbahasa Indonesia informal menjadi baris
pesanan terstruktur.

| | |
|---|---|
| F1 | 0.837 (81 pesan nyata, held-out per percakapan) |
| Ukuran | 11.00 MB (ONNX int8) |
| Latensi | 21.6 ms median, 1 thread |
| Inferensi | 100% offline, tanpa API, tanpa jaringan |
| Span palsu | 5 dari 41 pesan non-pesanan (12%) |

```bash
docker compose up --build
```

---

> **Dokumentasi lengkap saat ini hanya tersedia dalam bahasa Inggris:
> [README.md](README.md).** Versi Bahasa Indonesia akan ditulis ulang setelah
> pengerjaan fitur selesai, agar tidak menjadi dokumen yang isinya berbeda
> dengan versi utamanya.

| Dokumen | Isi |
|---|---|
| [README.md](README.md) | Arsitektur, keputusan desain, hasil pengukuran, batasan |
| [docs/PRD.md](docs/PRD.md) | Spesifikasi lengkap |
| [docs/RESULTS.md](docs/RESULTS.md) | Seluruh angka hasil pengukuran |
