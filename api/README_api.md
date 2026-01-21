##### Carpeta api/

Contiene el **backend del Web Service**.

##### 

##### 

##### Responsabilidad

* Exponer endpoints HTTP (FastAPI)
* Cargar el índice FAISS ya generado
* Consultar el LLM (Gemini / OpenRouter)
* NO genera embeddings

Esta carpeta es segura para despliegues livianos (sin PyTorch).

