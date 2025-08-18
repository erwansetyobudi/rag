# Single-Document RAG Chatbot — OpenAI API (tanpa llama.cpp)
# ----------------------------------------------------------------------------
# - Menggunakan OpenAI Chat API untuk generasi jawaban (RAG tetap: embedding & FAISS lokal).
# - Membaca API key dari environment variable: OPENAI_API_KEY.
# - UI Streamlit : upload PDF → tanya → jawab + sitasi [p.X].
# ----------------------------------------------------------------------------
# Cara jalan (Windows/Laragon):
# 1) Aktifkan venv:          .\.venv\Scripts\activate
# 2) Install deps:            pip install streamlit pypdf sentence-transformers faiss-cpu openai tiktoken scikit-learn
# 3) Set API key (sekali):    setx OPENAI_API_KEY "sk-PASTE_KEY_KAMU"
#    (Tutup terminal, buka lagi, lalu activate venv lagi)
# 4) Jalankan:                streamlit run rag-bot.py
# ----------------------------------------------------------------------------

import os
import io
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from pypdf import PdfReader

# -------------------- Index backends --------------------
try:
    import faiss  # type: ignore
    _has_faiss = True
except Exception:
    _has_faiss = False

try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    _has_sklearn = True
except Exception:
    _has_sklearn = False

from sentence_transformers import SentenceTransformer

# OpenAI SDK (>=1.0)
from openai import OpenAI

# --------------------- App Config ----------------------
st.set_page_config(page_title="RAG Chatbot (OpenAI)", layout="wide")
st.title("🔎📄 Single-Document RAG Chatbot — OpenAI API")
st.caption("Upload satu PDF jurnal → saya ekstrak, indeks, dan jawab pertanyaan berdasarkan konteks dari PDF.")

# ---------------------- Data Types ---------------------
@dataclass
class Chunk:
    page: int
    text: str
    vector: Optional[np.ndarray] = None

# ------------------------ Utils ------------------------

def read_pdf(file: io.BytesIO) -> List[Tuple[int, str]]:
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i, text))
    return pages


def normalize_whitespace(s: str) -> str:
    return " ".join(s.split())


def chunk_text(text: str, page: int, chunk_size: int = 1000, overlap: int = 200) -> List[Tuple[int, str]]:
    text = normalize_whitespace(text)
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append((page, chunk))
        if end == n:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks

@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# ------------------ Vector index builders ------------------

def build_index(embeddings: np.ndarray):
    if _has_faiss:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        return ("faiss", index)
    elif _has_sklearn:
        nn = NearestNeighbors(n_neighbors=10, metric="cosine")
        nn.fit(embeddings)
        return ("sklearn", nn)
    else:
        raise RuntimeError("Tidak ada backend index (FAISS atau scikit-learn) terpasang.")


def search_index(kind, index, query_emb: np.ndarray, k: int = 4):
    if kind == "faiss":
        q = query_emb.copy()
        faiss.normalize_L2(q)
        sims, idxs = index.search(q.astype("float32"), k)
        return idxs[0], sims[0]
    else:  # sklearn
        dists, idxs = index.kneighbors(query_emb, n_neighbors=k)
        sims = 1.0 - dists[0]
        return idxs[0], sims

# ---------------------- OpenAI backend ---------------------
SYSTEM_PROMPT = (
    "You are a helpful research assistant. Answer the user's question using ONLY the provided context. "
    "If the answer is not in the context, say you don't know. Use concise language and include citations as [p.X]."
)

@st.cache_resource(show_spinner=False)
def init_openai():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY tidak ditemukan. Set environment variable terlebih dahulu.")
    return OpenAI()


def build_prompt(question: str, store: List[Chunk], idxs) -> str:
    context_blocks = []
    for i in idxs:
        c = store[int(i)]
        context_blocks.append(f"[p.{c.page}]\n{c.text.strip()}")
    context = "\n\n".join(context_blocks)
    prompt = (
        f"System instructions:\n{SYSTEM_PROMPT}\n\n"
        f"Context from the paper (use citations like [p.N]):\n{context}\n\n"
        f"User question:\n{question}\n\n"
        f"Answer in Indonesian."
    )
    return prompt


def openai_generate(client: OpenAI, prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 512) -> str:
    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return (chat.choices[0].message.content or "").strip()

# ------------------------ Sidebar UI -----------------------
st.sidebar.header("Pengaturan")
chunk_size = st.sidebar.slider("Panjang chunk (karakter)", 500, 2000, 1000, 50)
overlap = st.sidebar.slider("Overlap (karakter)", 50, 400, 200, 10)
retrieval_k = st.sidebar.slider("Top-K Retrieval", 1, 10, 4, 1)

embed_model = st.sidebar.text_input("Embedding model (cached)", value="sentence-transformers/all-MiniLM-L6-v2")
openai_model = st.sidebar.text_input("OpenAI chat model", value="gpt-4o-mini")
max_new_tokens = st.sidebar.slider("Max tokens jawaban", 128, 2048, 512, 32)

st.sidebar.markdown("""
**Catatan:**
- Embedding berjalan lokal (offline setelah sekali unduh model embedding).
- Hanya proses generasi jawaban yang memakai OpenAI API.
""")

# --------------------------- App ---------------------------
upload = st.file_uploader("Unggah PDF jurnal (maks 30MB)", type=["pdf"], accept_multiple_files=False)

if upload is None:
    st.info("Unggah satu PDF untuk mulai.")
    st.stop()

# 1) Extract
with st.spinner("Mengekstrak teks dari PDF…"):
    pages = read_pdf(upload)
    total_chars = sum(len(t) for _, t in pages)
    st.success(f"Terbaca {len(pages)} halaman, ±{total_chars:,} karakter.")

# 2) Chunking
with st.spinner("Melakukan chunking…"):
    chunk_pairs: List[Tuple[int, str]] = []
    for pageno, text in pages:
        chunk_pairs.extend(chunk_text(text, page=pageno, chunk_size=chunk_size, overlap=overlap))
    texts = [t for _, t in chunk_pairs]
    meta_pages = [p for p, _ in chunk_pairs]
    st.write(f"Total chunk: {len(texts)}")

# 3) Embedding + Index
with st.spinner("Membangun index vektor…"):
    embedder = load_embedder(embed_model)
    embs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    kind, index = build_index(embs)
    store: List[Chunk] = [Chunk(page=meta_pages[i], text=texts[i], vector=embs[i]) for i in range(len(texts))]
    st.success(f"Index siap. Backend: {kind}")

# 4) Init OpenAI client
try:
    client = init_openai()
except Exception as e:
    st.error(str(e))
    st.stop()

# 5) Chat UI
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Pertanyaan Anda tentang jurnal:")
ask = st.button("Tanya")

if ask and question.strip():
    with st.spinner("Mencari konteks…"):
        q_emb = embedder.encode([question], convert_to_numpy=True)
        idxs, sims = search_index(kind, index, q_emb, k=retrieval_k)

    with st.spinner("Menghasilkan jawaban (OpenAI)…"):
        prompt = build_prompt(question, store, idxs)
        answer = openai_generate(client, prompt, model=openai_model, max_tokens=max_new_tokens)

    st.markdown("### Jawaban")
    st.write(answer)

    st.markdown("### Sumber konteks (Top-K)")
    for rank, i in enumerate(idxs, start=1):
        c = store[int(i)]
        with st.expander(f"Rank {rank} · Halaman {c.page}"):
            st.write(c.text)

    st.session_state.history.append((question, answer))

# 6) History
if st.session_state.history:
    st.markdown("---")
    st.markdown("### Riwayat Tanya-Jawab")
    for i, (q, a) in enumerate(reversed(st.session_state.history), start=1):
        st.markdown(f"**Q{i}.** {q}")
        st.markdown(f"**A{i}.** {a}")
