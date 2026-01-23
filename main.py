import os
import json
import time
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================================
# Config
# ==========================================================
STORE_DIR = os.environ.get("STORE_DIR", "/opt/render/project/src/rag_store")
META_PATH = os.path.join(STORE_DIR, "meta.json")
INDEX_PATH = os.path.join(STORE_DIR, "index.npy")  # si usás numpy index
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
GEN_MODEL = os.environ.get("GEN_MODEL", "mistralai/devstral-2512:free")

TOP_K = int(os.environ.get("TOP_K", "5"))

# ==========================================================
# FastAPI
# ==========================================================
app = FastAPI(title="RAG Chatbot API")

# ==========================================================
# ✅ CORS CORRECTO PARA GITHUB PAGES
# ==========================================================
ALLOWED_ORIGINS = [
    "https://rodrigo1964-ai.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,      # ✅ IMPORTANTE
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Models
# ==========================================================
class ChatRequest(BaseModel):
    question: str
    k: int = TOP_K

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]

# ==========================================================
# Utils
# ==========================================================
def store_status():
    meta_exists = os.path.exists(META_PATH)
    index_exists = os.path.exists(INDEX_PATH)
    meta_items = 0

    if meta_exists:
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta_items = len(meta) if isinstance(meta, list) else 0
        except:
            meta_items = 0

    return {
        "estado": "correcto",
        "indice_dim": 384,
        "meta_items": meta_items,
        "modelo_incrustado": EMBED_MODEL,
        "modelo_gen": GEN_MODEL,
        "directorio_tienda": STORE_DIR,
        "indice_tiene": index_exists,
        "meta_tiene": meta_exists,
    }

def require_store_files():
    if not os.path.exists(META_PATH):
        raise HTTPException(status_code=500, detail="meta.json no existe en rag_store")
    if not os.path.exists(INDEX_PATH):
        raise HTTPException(status_code=500, detail="index.npy no existe en rag_store")

# ==========================================================
# Routes
# ==========================================================
@app.get("/")
def root():
    return {"status": "ok", "service": "mi-web rag"}

@app.get("/health")
def health():
    return store_status()

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # si querés permitir funcionar aunque no estén los archivos, comentá esto
    require_store_files()

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Pregunta vacía")

    # ⚠️ ACA va tu lógica real de RAG
    # Por ahora devolvemos demo
    answer = f"Recibí tu pregunta: {question}\n(backend funcionando OK)"
    citations = []

    return {"answer": answer, "citations": citations}
