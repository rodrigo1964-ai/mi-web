El entorno rag_clean se crea con:
rag_clean.txt

El entorno rag_clean debe instalarse ademas:
>conda activate rag_clean
>pip install fastapi uvicorn

Debe ademas levantarse el backend; 
Levantar el backend (puerto 8001):
>cd C:\mi_html
>conda activate rag_clean
>cd C:\mi_html\api
>uvicorn main:app --host 127.0.0.1 --port 8001


En cmd rag_clean levantar el servidor:
>cd C:\mi_html
>conda activate rag_clean
>python -m http.server 8000


En el browser: 
http://localhost:8000
