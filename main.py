import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai


# ========= CONFIG =========
PDF_DIR = Path("./mis_pdf")         # tus PDFs
STORE_DIR = Path("./rag_store") # índice persistente
INDEX_PATH = STORE_DIR / "faiss.index"
META_PATH  = STORE_DIR / "chunks_meta.json"


STORE_DIR.mkdir(parents=True, exist_ok=True)



WORDS_PER_CHUNK = 450
OVERLAP = 60
EMBED_MODEL = "all-MiniLM-L6-v2"

GEN_MODEL = "models/gemini-2.5-flash"
# =========================


def leer_pdf_por_paginas(path: Path):
    reader = PdfReader(str(path))
    for page_num, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        if txt:
            yield page_num, txt


def chunk_text(text: str, words_per_chunk=450, overlap=60):
    words = text.split()
    chunks = []
    i = 0
    step = max(1, words_per_chunk - overlap)
    while i < len(words):
        chunk = " ".join(words[i:i+words_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def build_or_load_index():
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_PATH.exists() and META_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        return index, meta

    # construir desde PDFs
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No hay PDFs en {PDF_DIR}")

    meta = []
    all_chunks = []

    global_chunk_id = 0

    for pdf_path in pdf_files:
        for page_num, page_text in leer_pdf_por_paginas(pdf_path):
            page_chunks = chunk_text(page_text, WORDS_PER_CHUNK, OVERLAP)

            for local_id, ch in enumerate(page_chunks):
                global_chunk_id += 1
                meta.append({
                    "chunk_id": global_chunk_id,
                    "pdf": pdf_path.name,
                    "page": page_num,
                    "local_chunk": local_id + 1,
                    "text": ch
                })
                all_chunks.append(ch)

    embedder = SentenceTransformer(EMBED_MODEL)
    doc_vecs = embedder.encode(all_chunks, convert_to_numpy=True).astype("float32")

    dim = doc_vecs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(doc_vecs)

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return index, meta


# ========== FastAPI ==========
app = FastAPI(title="RAG Chatbot API")

# permitir llamadas desde tu página localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    k: int = 6


class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]


# cargar al inicio
index, meta = build_or_load_index()
embedder = SentenceTransformer(EMBED_MODEL)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Falta variable de entorno GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def search(query: str, k: int = 6):
    qv = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(qv, k)

    hits = []
    for idx, dist in zip(I[0], D[0]):
        if idx == -1:
            continue
        item = meta[int(idx)]
        hits.append((float(dist), item))


    return hits


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    hits = search(req.question, k=req.k)

    contexto = []
    citations = []

    for dist, item in hits:
        contexto.append(
            f"[{item['pdf']} p.{item['page']}]\n{item['text']}"
        )
        citations.append({
            "pdf": item["pdf"],
            "page": item["page"],
            "chunk_id": item["chunk_id"]
        })

    contexto = "\n\n".join(contexto)

    prompt = f"""
Sos un asistente técnico especializado en procedimientos metrológicos.
Respondé en español.

REGLAS:
- Usá SOLO el CONTEXTO.
- Si falta info, decí: "No está documentado en el corpus."
- Citá SIEMPRE así: (archivo p.X). Ej: (el-002e.pdf p.7)
- No inventes.

CONTEXTO:
{contexto}

PREGUNTA:
{req.question}
"""

    resp = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt
    )

    return {"answer": resp.text, "citations": citations}
