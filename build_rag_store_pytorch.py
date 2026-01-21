import json
from pathlib import Path

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ========= CONFIG =========
PDF_DIR = Path(r"C:\mi_html\mis_pdf")          # tus PDFs
STORE_DIR = Path(r"C:\mi_html\api\rag_store")  # carpeta destino del store
INDEX_PATH = STORE_DIR / "faiss.index"
META_PATH  = STORE_DIR / "chunks_meta.json"

WORDS_PER_CHUNK = 450
OVERLAP = 60
EMBED_MODEL = "all-MiniLM-L6-v2"
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


def build_store(force_rebuild: bool = False):
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    if not force_rebuild and INDEX_PATH.exists() and META_PATH.exists():
        print("[OK] Store ya existe. No se reconstruye.")
        print(f" - {INDEX_PATH}")
        print(f" - {META_PATH}")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No hay PDFs en: {PDF_DIR}")

    print(f"[INFO] PDFs encontrados: {len(pdf_files)}")
    meta = []
    all_chunks = []

    global_chunk_id = 0

    for pdf_path in pdf_files:
        print(f"[INFO] Procesando: {pdf_path.name}")
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

    print(f"[INFO] Total chunks: {len(all_chunks)}")
    embedder = SentenceTransformer(EMBED_MODEL)
    doc_vecs = embedder.encode(all_chunks, convert_to_numpy=True).astype("float32")

    dim = doc_vecs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(doc_vecs)

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Store generado correctamente:")
    print(f" - {INDEX_PATH}")
    print(f" - {META_PATH}")


if __name__ == "__main__":
    # Si querés forzar rebuild:
    # build_store(force_rebuild=True)
    build_store(force_rebuild=False)
