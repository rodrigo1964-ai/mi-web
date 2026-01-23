// ===============================
// Config backend (Render)
// ===============================

// Backend REAL (Render)
const API_URL = "https://mi-web-f295.onrender.com/chat";

// (Opcional) backend local para pruebas
// const API_URL = "http://localhost:8001/chat";

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

async function askBackend(question) {
  // ✅ CORRECTO: guardar la respuesta en r
  const r = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!r.ok) {
    const t = await r.text();
    throw new Error(`Backend error ${r.status}: ${t}`);
  }

  // ✅ CORRECTO
  return await r.json();
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  input.value = "";
  sendBtn.disabled = true;

  addMsg(question, "user");
  addMsg("Pensando...", "bot");

  try {
    const data = await askBackend(question);

    // limpiar "Pensando..."
    chat.lastChild.remove();

    // según tu API, puede venir como "answer"
    const answer = data.answer ?? JSON.stringify(data, null, 2);
    addMsg(answer, "bot");
  } catch (err) {
    chat.lastChild.remove();
    addMsg("⚠️ Error: " + err.message, "bot");
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});
