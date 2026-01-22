const chat = document.getElementById("chat");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("question");
const sendBtn = document.getElementById("sendBtn");

// Backend (Render)
const API_URL = "https://mi-web-f295.onrender.com/chat";



function addMessage(text, who="bot"){
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function askBackend(question){
  const r = await fetch(API_URL, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ question })
  });

  if(!r.ok){
    const t = await r.text();
    throw new Error(`Backend error ${r.status}: ${t}`);
  }
  return r.json();
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = questionInput.value.trim();
  if(!q) return;

  questionInput.value = "";
  questionInput.focus();

  addMessage(q, "user");
  sendBtn.disabled = true;

  // placeholder
  addMessage("Pensando...", "bot");
  const lastBot = chat.lastChild;

  try{
    const data = await askBackend(q);

    // JSON esperado: { answer: "...", citations: [{pdf,page},...] }
    let out = data.answer || "(sin respuesta)";
    if (Array.isArray(data.citations) && data.citations.length > 0){
      out += "\n\nFuentes:\n" + data.citations
        .map(c => `- ${c.pdf} p.${c.page}`)
        .join("\n");
    }

    lastBot.textContent = out;
  }catch(err){
    lastBot.textContent = "⚠️ Error: " + err.message;
  }finally{
    sendBtn.disabled = false;
  }
});

addMessage("Hola. Preguntame sobre los PDFs cargados.", "bot");
