// ===============================
// Config backend (auto local / Render)
// ===============================

const LOCAL_BACKEND = "http://127.0.0.1:8001";
const RENDER_BACKEND = "https://mi-web-f295.onrender.com";

// Si estoy en localhost → uso backend local.
// Si estoy en cualquier otro host → uso Render.
const BACKEND_BASE =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
    ? LOCAL_BACKEND
    : RENDER_BACKEND;

const API_URL = BACKEND_BASE + "/chat";
const HEALTH_URL = BACKEND_BASE + "/health";

// ===============================
// Mostrar backend activo en index.html
// ===============================
document.addEventListener("DOMContentLoaded", () => {
  const el = document.getElementById("backend-url");
  if (el) {
    el.textContent = API_URL;
  }
});

//==============================================================

const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const input = document.getElementById("question");
const sendBtn = document.getElementById("sendBtn");

function addMsg(text, who = "bot") {
  const div = document.createElement("div");
  div.className = "msg " + who;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

// ---- Util: timeout para fetch ----
function fetchWithTimeout(url, options = {}, timeoutMs = 65000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  return fetch(url, { ...options, signal: controller.signal })
    .finally(() => clearTimeout(timer));
}

// ---- Retry con backoff (Render free suele dormir) ----
async function askBackend(question, retries = 2) {
  const payload = { question, k: 4 };

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const r = await fetchWithTimeout(
        API_URL,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
        65000
      );

      if (!r.ok) {
        const t = await r.text();
        throw new Error(`Backend HTTP ${r.status}: ${t}`);
      }

      return await r.json();

    } catch (err) {
      const msg =
        err?.name === "AbortError"
          ? "Timeout esperando respuesta del backend (Render puede estar despertando)."
          : err.message;

      if (attempt < retries) {
        console.warn(`Intento ${attempt + 1} falló: ${msg}. Reintentando...`);
        await new Promise(res => setTimeout(res, 2500 * (attempt + 1)));
        continue;
      }

      throw new Error(msg);
    }
  }
}

//==============================================================

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  input.value = "";
  sendBtn.disabled = true;

  addMsg(question, "user");
  addMsg("Pensando... (si Render está dormido puede tardar ~50s)", "bot");

  try {
    const data = await askBackend(question, 2);

    // limpiar "Pensando..."
    chat.lastChild.remove();

    const answer = data.answer ?? JSON.stringify(data, null, 2);
    addMsg(answer, "bot");

    if (data.citations && Array.isArray(data.citations) && data.citations.length) {
      const citeText = data.citations
        .map(c => `• ${c.document ?? "doc"} pág ${c.page ?? "?"}`)
        .join("\n");
      addMsg("Citas:\n" + citeText, "bot");
    }

  } catch (err) {
    chat.lastChild.remove();
    addMsg("⚠️ Error: " + err.message, "bot");

    addMsg(
      "Tip: si esto pasa seguido, abrí primero /health para despertar Render: " +
      HEALTH_URL,
      "bot"
    );
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});
