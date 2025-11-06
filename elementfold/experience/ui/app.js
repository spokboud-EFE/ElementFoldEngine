/* ──────────────────────────────────────────────────────────────────────────
   ElementFold UI • app.js
   Small, narrative JavaScript with clear delimiters and comments.
   No frameworks; just fetch() to the same-origin server.py endpoints.

   Upgrades in this version:
     • Progressive “adapter panel” helpers (status/spec/tick/driver/delta).
     • Parses adapter tick lines for ℱ, z, A and surfaces them as badges
       if an element with id="predBadges" exists (optional).
     • Safer error handling + Ctrl+Enter to send /steer.
     • Works with your current HTML; extra controls are bound only if present.
   ────────────────────────────────────────────────────────────────────────── */

// ──────────────────────────────────────────────────────────────────────────
// Helpers: DOM bindings (graceful if optional nodes are absent)
// ──────────────────────────────────────────────────────────────────────────
const el = (id) => document.getElementById(id);
const on = (node, evt, fn) => { if (node) node.addEventListener(evt, fn); };

const statusEl      = el("status");
const btnHealth     = el("btnHealth");

// Inference panel
const btnInfer      = el("btnInfer");
const inferText     = el("inferText");
const strategySel   = el("strategy");
const tempInp       = el("temperature");
const topkInp       = el("topk");
const toppInp       = el("topp");
const inferTokens   = el("inferTokens");
const inferLedger   = el("inferLedger");

// Steering / adapters panel
const modalitySel   = el("modality");
const steerPrompt   = el("steerPrompt");
const btnSteer      = el("btnSteer");
const steerOut      = el("steerOut");
const audioRow      = el("audioRow");
const audioPlayer   = el("audioPlayer");

// Optional, progressive‑enhancement targets (only if in your HTML)
const btnSpec       = el("btnSpec");
const btnStatus     = el("btnStatus");
const btnTick1      = el("btnTick1");
const btnTick5      = el("btnTick5");
const btnHold       = el("btnHold");
const btnUp         = el("btnStepUp");
const btnDown       = el("btnStepDown");
const driverSel     = el("driverSelect");  // values: sim|null|live (live passthrough if not supported)
const deltaInp      = el("deltaValue");
const btnDeltaSet   = el("btnDeltaSet");
const predBadges    = el("predBadges");    // container for ℱ / z / A badges (optional)
const adapterStatus = el("adapterStatus"); // optional compact status line

// ──────────────────────────────────────────────────────────────────────────
// HTTP helpers
// ──────────────────────────────────────────────────────────────────────────
async function postJSON(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${text ? ` — ${text}` : ""}`);
  }
  return await res.json();
}

async function get(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res;
}

// ──────────────────────────────────────────────────────────────────────────
// Health
// ──────────────────────────────────────────────────────────────────────────
async function onHealth() {
  try {
    const res = await get("/health");
    statusEl.textContent = res.ok ? "status: ✓ healthy" : "status: ✖ unhealthy";
  } catch (e) {
    statusEl.textContent = `status: ✖ cannot reach /health (${e.message})`;
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Inference: POST /infer
// ──────────────────────────────────────────────────────────────────────────
async function onInfer() {
  const strategy    = strategySel?.value ?? "sample";
  const temperature = parseFloat((tempInp?.value || "1").trim());
  const top_k       = parseInt((topkInp?.value || "0").trim(), 10);
  const top_p       = parseFloat((toppInp?.value || "0").trim());

  const payload = {
    strategy,
    temperature: Number.isFinite(temperature) ? temperature : 1,
    top_k: Number.isFinite(top_k) && top_k > 0 ? top_k : null,
    top_p: Number.isFinite(top_p) && top_p > 0 && top_p < 1 ? top_p : null,
  };

  const text = (inferText?.value || "").trim();
  if (text.length > 0) payload.text = text;

  try {
    const out = await postJSON("/infer", payload);
    const toks = Array.isArray(out.tokens) ? out.tokens : [];
    const ledg = Array.isArray(out.ledger) ? out.ledger : [];

    const head = toks.slice(0, 64);
    inferTokens.textContent = JSON.stringify(head, null, 2) + (toks.length > 64 ? " …" : "");

    const mean = ledg.reduce((a, b) => a + b, 0) / Math.max(1, ledg.length);
    inferLedger.textContent = `${Number.isFinite(mean) ? mean.toFixed(6) : "n/a"}  (mean over ${ledg.length})`;
  } catch (e) {
    inferTokens.textContent = `error: ${e.message}`;
    if (inferLedger) inferLedger.textContent = "";
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Steering / Adapters: POST /steer
// ──────────────────────────────────────────────────────────────────────────
async function steer(modality, prompt) {
  const payload = { modality, prompt };
  return await postJSON("/steer", payload);
}

function renderSteerText(text) {
  // reset audio preview
  if (audioRow)  audioRow.style.display = "none";
  if (audioPlayer) audioPlayer.src = "";

  // show text
  steerOut.textContent = text;

  // parse predictions if present, surface as badges when predBadges exists
  tryParseAndRenderPredictions(text);

  // optional compact adapter status line (phase/κ/p½/ctrl) if you like
  if (adapterStatus) {
    const phase = (text.match(/phase=([A-Z]+)/) || [])[1] || "?";
    const kap   = parseFloat((text.match(/κ=([0-9.]+)/) || [])[1] || "NaN");
    const phalf = parseFloat((text.match(/p½=([0-9.]+)/) || [])[1] || "NaN");
    adapterStatus.textContent =
      `phase=${phase} • κ=${Number.isFinite(kap)?kap.toFixed(3):"?"} • p½=${Number.isFinite(phal f)?phal f.toFixed(3):"?"}`;
  }
}

function renderSteerPayload(out) {
  // language adapters usually return a string in out.output
  if (typeof out.output === "string") {
    renderSteerText(out.output);
    return;
  }
  // JSON-ish → pretty print
  steerOut.textContent = JSON.stringify(out.output, null, 2);

  // if audio data URL present, preview it
  const dataUrl = out?.output?.data_url || null;
  if (dataUrl && typeof dataUrl === "string" && dataUrl.startsWith("data:audio")) {
    if (audioPlayer) audioPlayer.src = dataUrl;
    if (audioRow) audioRow.style.display = "";
  }
}

async function onSteer() {
  const modality = modalitySel?.value || "language";
  const prompt   = (steerPrompt?.value || "tick 1").trim() || "tick 1";

  try {
    const out = await steer(modality, prompt);
    renderSteerPayload(out);
  } catch (e) {
    steerOut.textContent = `error: ${e.message}`;
    if (audioRow) audioRow.style.display = "none";
    if (audioPlayer) audioPlayer.src = "";
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Prediction badges (optional UI sugar)
//   Looks for lines like: “ℱ=0.312  z=1.37  A≈e^(−2ℱ)=0.54”
// ──────────────────────────────────────────────────────────────────────────
function tryParseAndRenderPredictions(text) {
  if (!predBadges) return;  // no-op if container absent

  const mF = text.match(/ℱ\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)/);
  const mz = text.match(/\bz\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)/);
  const mA = text.match(/\bA(?:≈|=)[^0-9\-+]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)/);

  const F = mF ? parseFloat(mF[1]) : null;
  const z = mz ? parseFloat(mz[1]) : null;
  const A = mA ? parseFloat(mA[1]) : null;

  if (F == null && z == null && A == null) {
    predBadges.innerHTML = ""; // nothing to show
    return;
  }

  // compose badges (keep simple, no styles assumed)
  const parts = [];
  if (F != null && Number.isFinite(F)) parts.push(`<span class="badge">ℱ=${F.toFixed(3)}</span>`);
  if (z != null && Number.isFinite(z)) parts.push(`<span class="badge">z=${z.toFixed(3)}</span>`);
  if (A != null && Number.isFinite(A)) parts.push(`<span class="badge">A≈e^(−2ℱ)=${A.toFixed(3)}</span>`);
  predBadges.innerHTML = parts.join(" ");
}

// ──────────────────────────────────────────────────────────────────────────
// Progressive “adapter panel” helpers (bound only if controls exist)
//   These simply send textual commands to the selected adapter.
// ──────────────────────────────────────────────────────────────────────────
async function runAdapterCmd(cmd) {
  const modality = modalitySel?.value || "resonator";
  try {
    const out = await steer(modality, cmd);
    renderSteerPayload(out);
  } catch (e) {
    steerOut.textContent = `error: ${e.message}`;
  }
}

async function onSpec()   { await runAdapterCmd("spec"); }     // falls back to help if unknown
async function onStatus() { await runAdapterCmd("status"); }
async function onTick1()  { await runAdapterCmd("tick 1"); }
async function onTick5()  { await runAdapterCmd("tick 5"); }
async function onHold()   { await runAdapterCmd("hold"); }
async function onUp()     { await runAdapterCmd("step up 1"); }
async function onDown()   { await runAdapterCmd("step down 1"); }

async function onDriverChange() {
  if (!driverSel) return;
  const choice = (driverSel.value || "").trim().toLowerCase();
  if (!choice) return;
  await runAdapterCmd(`driver ${choice}`); // sim|null|live (live becomes no-op if adapter lacks it)
}

async function onDeltaSet() {
  if (!deltaInp) return;
  const val = parseFloat((deltaInp.value || "").trim());
  if (!Number.isFinite(val)) {
    steerOut.textContent = "error: delta must be a number";
    return;
  }
  await runAdapterCmd(`delta ${val}`);
}

// ──────────────────────────────────────────────────────────────────────────
// Wire up events
// ──────────────────────────────────────────────────────────────────────────
on(btnHealth, "click", onHealth);

on(btnInfer , "click", onInfer);

on(btnSteer , "click", onSteer);
on(steerPrompt, "keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    onSteer();
  }
});

// Optional adapter quick‑controls (bind only if present in your HTML)
on(btnSpec   , "click", onSpec);
on(btnStatus , "click", onStatus);
on(btnTick1  , "click", onTick1);
on(btnTick5  , "click", onTick5);
on(btnHold   , "click", onHold);
on(btnUp     , "click", onUp);
on(btnDown   , "click", onDown);
on(driverSel , "change", onDriverChange);
on(btnDeltaSet, "click", onDeltaSet);

// Do an initial health ping on load
onHealth();
