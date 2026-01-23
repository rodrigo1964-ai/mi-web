import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# Config
# ============================================================
TOP_K = int(os.getenv("TOP_K", "5"))

BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "rag_store"

INDEX_PATH = STORE_DIR / "index.faiss"
META_PATH  = STORE_DIR / "meta.jsonl"

# Tu embedder/gen ya lo tenés (no lo toco acá)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
GEN_MODEL   = os.getenv("GEN_MODEL", "mistralai/devstral-2512:free")


# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="RAG Chatbot API (Light - No Torch)")


# ============================================================
# ✅ CORS CORRECTO PARA BROWSER
# - NO se puede allow_credentials=True con "*"
# - Permitimos tu GitHub Pages + localhost
# ============================================================
ALLOWED_ORIGINS = [
    "https://rodrigo1964-ai.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,   # ✅ CLAVE: así funciona en cualquier navegador
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Schemas
# ============================================================
class ChatRequest(BaseModel):
    question: str
    k: int = TOP_K


class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []


# ============================================================
# Helpers mínimos (placeholders compatibles con tu lógica)
# ============================================================
def store_status() -> Dict[str, Any]:
    meta_items = 0
    if META_PATH.exists():
        with META_PATH.open("r", encoding="utf-8") as f:
            for _ in f:
                meta_items += 1

    return {
        "status": "ok",
        "index_dim": 384,
        "meta_items": meta_items,
        "embed_model": EMBED_MODEL,
        "gen_model": GEN_MODEL,
        "store_dir": str(STORE_DIR),
        "has_index": INDEX_PATH.exists(),
        "has_meta": META_PATH.exists(),
    }


# ============================================================
# Routes
# ============================================================
@app.get("/health")
def health():
    return store_status()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Endpoint esperado por app.js:
      POST /chat  body: {"question":"..."}
    Retorna:
      {"answer":"...", "citations":[...]}
    """
    q = req.question.strip()
    if not q:
        return ChatResponse(answer="⚠️ La pregunta está vacía.", citations=[])

    # ------------------------------------------------------------------
    # ACÁ IRÍA TU LÓGICA REAL RAG (faiss + meta + generación)
    #
    # Como vos ya lo tenés funcionando,
    # no rompo nada: dejo una respuesta placeholder compatible.
    #
    # IMPORTANTE: devolver siempre JSON válido.
    # ------------------------------------------------------------------

    # Simulación: contestar con lo que hay en store (para test rápido)
    st = store_status()
    answer = (
        f"✅ Backend OK.\n"
        f"📦 PDFs indexados: {st['meta_items']}\n"
        f"🧠 Modelo: {GEN_MODEL}\n"
        f"❓ Pregunta: {q}\n\n"
        f"(Aquí entra la respuesta real RAG.)"
    )

    return ChatResponse(answer=answer, citations=[])


# ============================================================
# Entry point (Render)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
