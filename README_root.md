# Proyecto RAG – Estructura General

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** con una clara separación entre:
- procesamiento local pesado (embeddings)
- despliegue liviano en servidores (Render, Oracle, etc.)

## Estructura
- `api/` : backend FastAPI
- `mis_pdf/` : PDFs fuente
- `rag_store/` : embeddings e índice FAISS precomputados
- `main.py` : entrypoint del servicio
