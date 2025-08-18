# Single‑Document RAG Chatbot (Windows + Laragon + OpenAI API)
# By Erwan Setyo Budi

Chatbot “ala SciSpace” untuk **satu dokumen PDF**: ekstrak teks → chunking → embedding → **retrieval** (FAISS) → **jawaban** via OpenAI Chat API dengan sitasi halaman `[p.X]`.

## ✨ Fitur

* Upload **1 PDF** (jurnal ilmiah).
* **Chunking + overlap** yang dapat diatur.
* **Embedding lokal** (sentence-transformers) + **FAISS** (fallback ke scikit‑learn bila FAISS tidak tersedia).
* Jawaban memakai **OpenAI Chat API** (`gpt‑4o‑mini` default).
* Sitasi halaman otomatis dalam format `[p.X]`.
* UI **Streamlit** simpel.

---

## 🧱 Struktur Proyek

```
your-repo/
├─ rag-bot.py               # aplikasi Streamlit (versi OpenAI API)
├─ models/                  # (opsional, kosong – hanya bila kamu pakai versi offline)
├─ README.md
└─ requirements.txt         # opsional
```

**requirements.txt (opsional):**

```
streamlit
pypdf
sentence-transformers
faiss-cpu
openai
tiktoken
scikit-learn
```

---

## 🔧 Prasyarat

* **Windows 64-bit**
* **Laragon** (pakai Terminal Laragon / Cmder bawaan juga ok)
* **Python 3.10+ 64-bit** (yang dipakai Laragon)
* **Akun OpenAI** + **API Key** aktif (pastikan billing/kuota tersedia)

> Jangan pernah commit/unggah API key ke GitHub.

---

## 🚀 Quickstart (Terminal Laragon)

1. **Clone** repo ini (atau salin file `rag-bot.py` ke folder kerja):

```bat
cd C:\laragon\www
git clone <url-repo-kamu> rag-bot
cd rag-bot
```

2. **Buat & aktifkan virtual environment**

```bat
python -m venv .venv
.\.venv\Scripts\activate
```

3. **Install dependensi**

```bat
pip install --upgrade pip
pip install -r requirements.txt
:: atau:
:: pip install streamlit pypdf sentence-transformers faiss-cpu openai tiktoken scikit-learn
```

4. **Set API Key (sesi ini saja)**

```bat
set OPENAI_API_KEY=sk-PASTE_KEY_ANDA
```

(opsional, permanen untuk sesi mendatang)

```bat
setx OPENAI_API_KEY "sk-PASTE_KEY_ANDA"
:: lalu tutup terminal, buka lagi, dan activate venv ulang
```

5. **Jalankan aplikasi**

```bat
streamlit run rag-bot.py
```

6. **Buka browser** → `http://localhost:8501`

* Upload **PDF** (maks 30MB).
* Atur **chunk size**, **overlap**, dan **Top‑K** bila perlu.
* Isi pertanyaan → klik **Tanya**.
* Lihat jawaban + potongan konteks (expandable) beserta sitasi `[p.X]`.

---

## ⚙️ Konfigurasi di Sidebar

* **Panjang chunk**: default 1000 chars (besarkan untuk paragraf panjang).
* **Overlap**: default 200 chars (pastikan `< chunk`).
* **Top‑K Retrieval**: default 4 (naikkan jika pertanyaan kompleks).
* **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (diunduh sekali → bisa offline).
* **OpenAI chat model**: `gpt-4o-mini` (hemat biaya, bagus untuk RAG).
* **Max tokens jawaban**: batas output model.

---

## 🧪 Cara Kerja Singkat (RAG)

1. **Ekstrak PDF** per halaman.
2. **Chunking** dengan overlap + simpan metadata nomor halaman.
3. **Embedding** setiap chunk → **FAISS index**.
4. **Retrieval** Top‑K chunk relevan untuk pertanyaan.
5. **Prompting** ke OpenAI Chat API dengan konteks + instruksi “jawab hanya dari konteks”.
6. **Jawaban** + sitasi `[p.N]`.

---

## 🩹 Troubleshooting

### `OPENAI_API_KEY tidak ditemukan`

* Kamu membuka app di terminal yang **belum** memuat variabel.
  Solusi:

  ```bat
  .\.venv\Scripts\activate
  set OPENAI_API_KEY=sk-KEY
  streamlit run rag-bot.py
  ```

  Untuk permanen: `setx OPENAI_API_KEY "sk-KEY"` lalu **buka terminal baru**.

### `openai.RateLimitError: insufficient_quota (429)`

* Kuota habis / belum set **billing**.
  Solusi: aktifkan metode pembayaran atau ganti API key akun yang ada saldonya.

### Lambat / token boros

* Turunkan **Max tokens jawaban**.
* Pertajam pertanyaan (sebut bagian: “metodologi”, “hasil”, “batasan”).
* Kurangi **Top‑K** jika tidak perlu.

### FAISS tidak terpasang

* App akan **fallback** ke scikit‑learn otomatis. Jika ingin FAISS, pastikan sukses terinstall (`faiss-cpu`).

### MemoryError saat chunking (versi lama)

* Fungsi `chunk_text` di repo ini sudah mengandung *guard* supaya loop tidak macet. Pastikan pakai file terbaru.

---

## 🔐 Keamanan API Key

* Simpan di **environment variable** saja.
* **Jangan** commit key ke Git.
* Jika tak sengaja tersebar, **revoke/regenerate** di dashboard OpenAI.

---

## 🗺️ Roadmap (opsional)

* Input API key dari **sidebar** (tanpa set env).
* Export **Q\&A** ke CSV/Markdown.
* Mode **offline** (llama.cpp / Ollama) untuk tanpa internet.

---

## 📝 Lisensi

Silakan pilih lisensi yang sesuai (MIT / Apache‑2.0).

---

## 🙌 Kredit

* Streamlit, FAISS, sentence-transformers.
* OpenAI API untuk generasi jawaban.

---

### Skrinsut

![UI](docs/screenshot.png) *(opsional)*

---
