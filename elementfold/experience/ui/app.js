/* ──────────────────────────────────────────────────────────────────────────
   ElementFold UI • app.js
   Small, narrative JavaScript with clear delimiters and comments.
   No frameworks; just fetch() to the same-origin server.py endpoints.
   ────────────────────────────────────────────────────────────────────────── */

// ──────────────────────────────────────────────────────────────────────────
// Helpers: DOM bindings
// ──────────────────────────────────────────────────────────────────────────
const el = (id) => document.getElementById(id);
const statusEl      = el("status");
const btnHealth     = el("btnHealth");
const btnInfer      = el("btnInfer");
const btnSteer      = el("btnSteer");
const inferText     = el("inferText");
const strategySel   = el("strategy");
const tempInp       = el("temperature");
const topkInp       = el("topk");
const toppInp       = el("topp");
const inferTokens   = el("inferTokens");
const inferLedger   = el("inferLedger");
const modalitySel   = el("modality");
const steerPrompt   = el("steerPrompt");
const steerOut      = el("steerOut");
const audioRow      = el("audioRow");
const audioPlayer   = el("audioPlayer");

// ──────────────────────────────────────────────────────────────────────────
async function postJSON(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return await res.json();
}

// ──────────────────────────────────────────────────────────────────────────
// Health check
// ──────────────────────────────────────────────────────────────────────────
async function onHealth() {
  try {
    const res = await fetch("/health");
    statusEl.textContent = res.ok ? "status: ✓ healthy" : "status: ✖ unhealthy";
  } catch (e) {
    statusEl.textContent = "status: ✖ cannot reach /health";
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Inference: build payload, POST /infer, render small preview
// ──────────────────────────────────────────────────────────────────────────
async function onInfer() {
  // Build request body
  const strategy    = strategySel.value;
  const temperature = parseFloat(tempInp.value || "1");
  const top_k       = parseInt(topkInp.value || "0", 10);
  const top_p       = parseFloat(toppInp.value || "0");

  const payload = {
    strategy,
    temperature,
    top_k: top_k > 0 ? top_k : null,
    top_p: (top_p > 0 && top_p < 1) ? top_p : null,
  };

  // If user typed a prompt, let server tokenize it; else leave x=null for random
  const text = (inferText.value || "").trim();
  if (text.length > 0) payload.text = text;

  // Request → response
  try {
    const out = await postJSON("/infer", payload);
    // tokens: list[int], ledger: list[float]
    const head = (out.tokens || []).slice(0, 64);
    inferTokens.textContent = JSON.stringify(head, null, 2) + (out.tokens.length > 64 ? " …" : "");
    const mean = (out.ledger || []).reduce((a, b) => a + b, 0) / Math.max(1, (out.ledger || []).length);
    inferLedger.textContent = `${mean.toFixed(6)}  (mean over ${out.ledger?.length ?? 0})`;
  } catch (e) {
    inferTokens.textContent = `error: ${e.message}`;
    inferLedger.textContent = "";
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Steering: POST /steer {prompt, modality}, render string or JSON, audio preview
// ──────────────────────────────────────────────────────────────────────────
async function onSteer() {
  const modality = modalitySel.value;
  const prompt   = (steerPrompt.value || "").trim();

  try {
    const out = await postJSON("/steer", { prompt, modality });

    // Reset audio preview unless we detect audio payload
    audioRow.style.display = "none";
    audioPlayer.src = "";

    // language adapter → string; audio/multimodal → dict
    if (typeof out.output === "string") {
      steerOut.textContent = out.output;
      return;
    }

    // JSON-ish → pretty print
    steerOut.textContent = JSON.stringify(out.output, null, 2);

    // If audio adapter returned a data URL, preview it
    const dataUrl = out?.output?.data_url || null;
    if (dataUrl && typeof dataUrl === "string" && dataUrl.startsWith("data:audio")) {
      audioPlayer.src = dataUrl;
      audioRow.style.display = "";
    }
  } catch (e) {
    steerOut.textContent = `error: ${e.message}`;
    audioRow.style.display = "none";
    audioPlayer.src = "";
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Wire up events
// ──────────────────────────────────────────────────────────────────────────
btnHealth.addEventListener("click", onHealth);
btnInfer .addEventListener("click", onInfer);
btnSteer .addEventListener("click", onSteer);

// Do an initial health ping on load
onHealth();
