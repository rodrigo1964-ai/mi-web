import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import faiss
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===============================
# Config
# ===============================

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "rag_store"

FAISS_PATH = STORE_DIR / "faiss.index"
META_PATH = STORE_DIR / "chunks_meta.json"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
GEN_MODEL = os.getenv("GEN_MODEL", "mistralai/devstral-2512:free")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ===============================
# FastAPI
# ===============================

app = FastAPI(title="mi-web RAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego cerramos a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Modelos
# ===============================

class ChatRequest(BaseModel):
    question: str
    k: int = 5

class ChatResponse(BaseModel):
    answer: str
    hits: List[Dict[str, Any]]
    citations: List[Dict[str, Any]] = []

# ===============================
# Carga del store
# ===============================

def load_store():
    if not STORE_DIR.exists():
        raise RuntimeError(f"rag_store NO existe: {STORE_DIR}")
    if not FAISS_PATH.exists():
        raise RuntimeError(f"Falta faiss.index: {FAISS_PATH}")
    if not META_PATH.exists():
        raise RuntimeError(f"Falta chunks_meta.json: {META_PATH}")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    index = faiss.read_index(str(FAISS_PATH))
    return index, meta

# ===============================
# Embeddings dummy (demo)
# ===============================

def embed_text(text: str, dim: int) -> np.ndarray:
    v = np.zeros((dim,), dtype="float32")
    h = abs(hash(text)) % dim
    v[h] = 1.0
    return v

def search(index, meta: List[Dict[str, Any]], question: str, k: int):
    dim = index.d
    qv = embed_text(question, dim).reshape(1, -1)

    D, I = index.search(qv, k)

    hits = []
    for j in range(len(I[0])):
        idx = int(I[0][j])
        if 0 <= idx < len(meta):
            hits.append({"score": float(D[0][j]), **meta[idx]})
    return hits

# ===============================
# LLM (OpenRouter)
# ===============================

def call_openrouter(question: str, context: str) -> str:
    if not OPENROUTER_API_KEY:
        return "⚠️ Falta configurar OPENROUTER_API_KEY en Render."

    system = (
        "Sos un asistente técnico. Respondé en español. "
        "Usá ÚNICAMENTE el CONTEXTO provisto. "
        "Si no hay información suficiente, decí: 'No lo encuentro en los PDFs indexados.' "
        "Al final agregá una sección 'Citas:' con PDF y página."
    )

    payload = {
        "model": GEN_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}"}
        ],
        "temperature": 0.2,
        "max_tokens": 600
    }

    r = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ===============================
# Endpoints
# ===============================

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API en funcionamiento"}

@app.get("/health")
def health():
    info = {
        "status": "ok",
        "store_dir": str(STORE_DIR),
        "faiss_existe": FAISS_PATH.exists(),
        "chunks_meta_existe": META_PATH.exists(),
        "embed_model": EMBED_MODEL,
        "gen_model": GEN_MODEL,
        "openrouter_key": bool(OPENROUTER_API_KEY),
    }
    return info

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    t0 = time.time()

    index, meta = load_store()
    hits = search(index, meta, req.question, req.k)

    # ---- construir contexto desde hits ----
    context_blocks = []
    citations = []
    for h in hits:
        text = h.get("text") or h.get("chunk") or h.get("content") or ""
        doc = h.get("document") or h.get("pdf") or h.get("source") or "?"
        page = h.get("page") or h.get("pageno") or h.get("pagina") or "?"
        context_blocks.append(f"[{doc} pág {page}]\n{text}")
        citations.append({"document": doc, "page": page})

    context_text = "\n\n---\n\n".join(context_blocks)[:12000]  # límite razonable

    # ---- generar respuesta ----
    answer = call_openrouter(req.question, context_text)

    dt = round(time.time() - t0, 3)
    answer += f"\n\n⏱️ Tiempo total: {dt}s"

    return {"answer": answer, "hits": hits, "citations": citations}
