# Chatbot RAG + Frontend estático

Este proyecto es una demo simple de un **chatbot RAG** (Retrieval-Augmented Generation) que responde preguntas usando información extraída de **PDFs propios**.

- El **backend** es una API en **FastAPI** (Python) que sirve:
  - Búsqueda en un índice FAISS (`rag_store`).
  - Llamadas a un modelo de lenguaje vía **OpenRouter**.
- El **frontend** es una página estática con HTML + JS:
  - Se puede publicar en **GitHub Pages**.
  - Detecta automáticamente si está en `localhost` o en producción (Render) para elegir el backend.

---

## Arquitectura

- **Backend** (`main.py`)
  - Framework: FastAPI + Uvicorn.
  - Endpoints principales:
    - `GET /` → estado simple.
    - `GET /health` → información del índice y de la configuración.
    - `POST /chat` → endpoint del chatbot.
  - Usa un índice FAISS y un archivo `chunks_meta.json` dentro de `rag_store/`.

- **Frontend**
  - `index.html` → página principal, muestra contenido y el chat.
  - `app.js` → lógica del chat (envía la pregunta al backend y muestra respuestas).
  - (Opcional) `styles.css` → estilos de la página.

- **RAG store**
  - Carpeta `rag_store/`
    - `faiss.index` → índice FAISS con embeddings.
    - `chunks_meta.json` → metadatos de cada chunk (texto, documento, página, etc.).

---

## Dependencias principales (backend)

En general vas a necesitar algo como:

```bash
pip install fastapi uvicorn[standard] requests numpy faiss-cpu
