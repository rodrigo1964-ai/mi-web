import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import faiss

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===============================
# Config
# ===============================

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "rag_store"

FAISS_PATH = STORE_DIR / "faiss.index"
META_PATH = STORE_DIR / "chunks_meta.json"   # ✅ el nombre correcto

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
GEN_MODEL = os.getenv("GEN_MODEL", "mistralai/devstral-2512:free")

# ===============================
# FastAPI
# ===============================

app = FastAPI(title="mi-web RAG API", version="1.0")

# ✅ CORS para GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # después lo cerramos a tu dominio si querés
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
# ⚠️ Esto está para que no reviente.
# Luego se reemplaza por embeddings reales.
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
    }

    try:
        if FAISS_PATH.exists():
            index = faiss.read_index(str(FAISS_PATH))
            info["index_dim"] = int(index.d)
        else:
            info["index_dim"] = None
    except Exception as e:
        info["index_dim_error"] = str(e)

    try:
        if META_PATH.exists():
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            info["meta_items"] = len(meta)
        else:
            info["meta_items"] = 0
    except Exception as e:
        info["meta_items_error"] = str(e)

    return info

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    t0 = time.time()

    index, meta = load_store()
    hits = search(index, meta, req.question, req.k)

    answer = (
        f"Backend OK.\n"
        f"PDFs indexados: {len(meta)}\n"
        f"Modelo: {GEN_MODEL}\n"
        f"Pregunta: {req.question}\n\n"
        f"(Aquí entra la respuesta real RAG.)"
    )

    dt = round(time.time() - t0, 3)
    answer += f"\n\n⏱️ Tiempo: {dt}s"

    return {"answer": answer, "hits": hits}
