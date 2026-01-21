import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from fastembed import TextEmbedding

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai


# ================= CONFIG =================
# Para que funcione en .py y también en Jupyter
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

STORE_DIR = BASE_DIR / "rag_store"
INDEX_PATH = STORE_DIR / "faiss.index"
META_PATH  = STORE_DIR / "chunks_meta.json"

TOP_K = int(os.environ.get("TOP_K", "6"))

# ⚠️ Tiene que ser el mismo modelo usado en build_rag_store_fastembed.py
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Modelo Gemini para generar la respuesta (NO embeddings)
GEN_MODEL = os.environ.get("GEMINI_GEN_MODEL", "models/gemini-2.5-flash")
# =========================================


# ========== FastAPI ==========
app = FastAPI(title="RAG Chatbot API (Light - No Torch)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # si querés lo restringimos después
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    k: int = TOP_K


class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]


def require_store_files():
    if not INDEX_PATH.exists():
        raise RuntimeError(f"Falta {INDEX_PATH}")
    if not META_PATH.exists():
        raise RuntimeError(f"Falta {META_PATH}")


def load_store() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    require_store_files()
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return index, meta


def init_embedder() -> TextEmbedding:
    # ONNX (no torch)
    return TextEmbedding(model_name=EMBED_MODEL)


def embed_query(embedder: TextEmbedding, text: str) -> np.ndarray:
    vec = next(embedder.embed([text]))
    return np.array(vec, dtype="float32").reshape(1, -1)


def search(index: faiss.Index, meta: List[Dict[str, Any]], embedder: TextEmbedding, query: str, k: int):
    qv = embed_query(embedder, query)
    D, I = index.search(qv, k)

    hits = []
    for j in range(min(k, I.shape[1])):
        idx = int(I[0][j])
        if idx < 0 or idx >= len(meta):
            continue
        hits.append((float(D[0][j]), meta[idx]))
    return hits


def get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta variable de entorno GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


# ======= Load global resources =======
index, meta = load_store()
embedder = init_embedder()
client = get_gemini_client()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_dim": index.d,
        "meta_items": len(meta),
        "embed_model": EMBED_MODEL,
        "gen_model": GEN_MODEL,
        "store_dir": str(STORE_DIR),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    hits = search(index, meta, embedder, req.question, k=req.k)

    contexto = []
    citations = []

    for dist, item in hits:
        contexto.append(f"[{item['pdf']} p.{item['page']}]\n{item['text']}")
        citations.append({
            "pdf": item["pdf"],
            "page": item["page"],
            "chunk_id": item.get("chunk_id"),
        })

    contexto_txt = "\n\n".join(contexto)

    prompt = f"""
Sos un asistente técnico especializado en procedimientos metrológicos.
Respondé en español.

REGLAS:
- Usá SOLO el CONTEXTO.
- Si falta info, decí: "No está documentado en el corpus."
- Citá SIEMPRE así: (archivo p.X). Ej: (el-002e.pdf p.7)
- No inventes.

CONTEXTO:
{contexto_txt}

PREGUNTA:
{req.question}
""".strip()

    resp = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt
    )

    return {"answer": resp.text, "citations": citations}
