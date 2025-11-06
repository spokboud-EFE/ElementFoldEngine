/* ──────────────────────────────────────────────────────────────────────────
   ElementFold UI • app.js
   Small, narrative JavaScript with clear delimiters and comments.
   No frameworks; just fetch() to the same-origin server.py endpoints.

   Upgrades in this version:
     • Progressive “adapter panel” helpers (status/spec→help/tick/driver/delta).
     • Parses adapter tick lines for ℱ, z, A and surfaces them as badges
       if an element with id="predBadges" exists (optional).
     • Safer error handling + Ctrl+Enter to send /steer and /infer.
     • Optional "relax" block for /infer if advanced inputs are present.
     • NEW: telemetry polling for /telemetry/state and /telemetry/counsel,
       with caption, bullets, numbers snapshot, and clickable next_actions.
     • Auto-populate adapter list from GET /adapters (if <select id="modality"> exists).
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

// (Optional) Relaxation inputs — only used if present
const relaxEta      = el("relaxEta");        // number
const relaxW        = el("relaxWeight");     // number (eta_path_weight)
const relaxRho      = el("relaxRho");        // number
const relaxLambda   = el("relaxLambda");     // number
const relaxD        = el("relaxD");          // number
const relaxPhiInf   = el("relaxPhiInf");     // number
const relaxSteps    = el("relaxSteps");      // integer
const relaxDt       = el("relaxDt");         // number

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
const driverSel     = el("driverSelect");  // values: sim|null|live (live becomes no-op if not supported)
const deltaInp      = el("deltaValue");
const btnDeltaSet   = el("btnDeltaSet");
const predBadges    = el("predBadges");    // container for ℱ / z / A badges (optional)
const adapterStatus = el("adapterStatus"); // optional compact status line

// NEW: Telemetry/Counsel panel (all optional)
const teleCaption   = el("teleCaption");   // one-line caption from /telemetry/counsel
const teleBullets   = el("teleBullets");   // <ul> container for bullet <li>
const teleNumbers   = el("teleNumbers");   // <pre> or <code> to show numbers snapshot
const teleActions   = el("teleActions");   // container for "chips" (next_actions)
const teleConfidence= el("teleConfidence");// small text: confidence + band
const teleTs        = el("teleTs");        // last fetch timestamp
const teleLLM       = el("teleLLM");       // shows "LLM: on/off"
const btnTeleStart  = el("btnTeleStart");
const btnTeleStop   = el("btnTeleStop");
const teleEveryMs   = el("teleEveryMs");   // <input type=number> poll ms

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

async function getJSON(path) {
  const res = await get(path);
  return await res.json();
}

// ──────────────────────────────────────────────────────────────────────────
// Health
// ──────────────────────────────────────────────────────────────────────────
async function onHealth() {
  try {
    const body = await getJSON("/health");
    const dev = body?.device || "unknown";
    const ready = body?.model_ready ? "✓ model" : "… training on demand";
    statusEl && (statusEl.textContent = `status: ✓ healthy • ${dev} • ${ready}`);
  } catch (e) {
    statusEl && (statusEl.textContent = `status: ✖ cannot reach /health (${e.message})`);
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Relaxation payload builder (optional panel)
// ──────────────────────────────────────────────────────────────────────────
function buildRelaxPayloadIfAny() {
  // If no relax fields exist, don’t send anything.
  if (!relaxEta && !relaxW && !relaxRho && !relaxLambda && !relaxD && !relaxPhiInf && !relaxSteps && !relaxDt) {
    return null;
  }

  // Helper: parse number safely
  const num = (inp) => {
    if (!inp) return null;
    const v = parseFloat((inp.value || "").trim());
    return Number.isFinite(v) ? v : null;
  };
  const int = (inp) => {
    if (!inp) return null;
    const v = parseInt((inp.value || "").trim(), 10);
    return Number.isFinite(v) ? v : null;
  };

  const relax = {};
  const eta    = num(relaxEta);
  const w      = num(relaxW);
  const rho    = num(relaxRho);
  const lam    = num(relaxLambda);
  const D      = num(relaxD);
  const phiInf = num(relaxPhiInf);
  const steps  = int(relaxSteps);
  const dt     = num(relaxDt);

  if (eta   != null) relax.eta = eta;
  if (w     != null) relax.eta_path_weight = w;
  if (rho   != null) relax.rho = rho;
  if (lam   != null) relax["lambda"] = lam;
  if (D     != null) relax.D = D;
  if (phiInf!= null) relax.phi_inf = phiInf;
  if (steps != null && steps >= 1) relax.steps = steps;
  if (dt    != null) relax.dt = dt;

  // If nothing valid was provided, return null to avoid sending an empty object.
  return Object.keys(relax).length ? relax : null;
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

  // Optional: include relaxation block if advanced inputs exist
  const relax = buildRelaxPayloadIfAny();
  if (relax) payload.relax = relax;

  try {
    const out = await postJSON("/infer", payload);
    const toks = Array.isArray(out.tokens) ? out.tokens : [];
    const ledg = Array.isArray(out.ledger) ? out.ledger : [];

    const head = toks.slice(0, 64);
    if (inferTokens) {
      inferTokens.textContent = JSON.stringify(head, null, 2) + (toks.length > 64 ? " …" : "");
    }

    if (inferLedger) {
      const mean = ledg.reduce((a, b) => a + b, 0) / Math.max(1, ledg.length);
      inferLedger.textContent = `${Number.isFinite(mean) ? mean.toFixed(6) : "n/a"}  (mean over ${ledg.length})`;
    }
  } catch (e) {
    if (inferTokens) inferTokens.textContent = `error: ${e.message}`;
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
  if (audioRow)    audioRow.style.display = "none";
  if (audioPlayer) audioPlayer.src = "";

  // show text
  if (steerOut) steerOut.textContent = text;

  // parse predictions if present, surface as badges when predBadges exists
  tryParseAndRenderPredictions(text);

  // optional compact adapter status line (phase/κ/p½/ctrl) if you like
  if (adapterStatus) {
    const phase = (text.match(/phase=([A-Z]+)/) || [])[1] || "?";
    const kap   = parseFloat((text.match(/κ=([0-9.]+)/) || [])[1] || "NaN");
    const phalf = parseFloat((text.match(/p½=([0-9.]+)/) || [])[1] || "NaN");
    const mCtrl = text.match(/ctrl=\{β:([0-9.]+),\s*γ:([0-9.]+),\s*⛔:([0-9.]+)\}/);
    const beta  = mCtrl ? parseFloat(mCtrl[1]) : NaN;
    const gamma = mCtrl ? parseFloat(mCtrl[2]) : NaN;
    const clamp = mCtrl ? parseFloat(mCtrl[3]) : NaN;

    const base = `phase=${phase} • κ=${Number.isFinite(kap) ? kap.toFixed(3) : "?"} • p½=${Number.isFinite(phalf) ? phalf.toFixed(3) : "?"}`;
    const ctrl = mCtrl ? ` • β=${beta.toFixed(2)} γ=${gamma.toFixed(2)} ⛔=${clamp.toFixed(1)}` : "";
    adapterStatus.textContent = base + ctrl;
  }
}

function renderSteerPayload(out) {
  // language adapters usually return a string in out.output
  if (typeof out.output === "string") {
    renderSteerText(out.output);
    return;
  }
  // JSON-ish → pretty print
  if (steerOut) steerOut.textContent = JSON.stringify(out.output, null, 2);

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
    if (steerOut) steerOut.textContent = `error: ${e.message}`;
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
    if (steerOut) steerOut.textContent = `error: ${e.message}`;
  }
}

// Note: resonator doesn’t implement 'spec', so map it to 'help' for now.
async function onSpec()   { await runAdapterCmd("help"); }
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
    if (steerOut) steerOut.textContent = "error: delta must be a number";
    return;
  }
  await runAdapterCmd(`delta ${val}`);
}

// ──────────────────────────────────────────────────────────────────────────
// Telemetry: polling /telemetry/state and /telemetry/counsel
// ──────────────────────────────────────────────────────────────────────────
let _teleTimer = null;
let _teleBusy = false;

function setTeleStatus(ts, llmUsed, conf) {
  if (teleTs)        teleTs.textContent = ts ? new Date(ts * 1000).toLocaleTimeString() : "";
  if (teleLLM)       teleLLM.textContent = llmUsed ? "LLM: on" : "LLM: off";
  if (teleConfidence) teleConfidence.textContent = (typeof conf === "number")
      ? `confidence: ${(conf * 100).toFixed(0)}%`
      : "";
}

function renderCounsel(c) {
  if (!c || typeof c !== "object") return;
  if (teleCaption)  teleCaption.textContent = c.caption || "—";
  if (teleNumbers)  teleNumbers.textContent = JSON.stringify(c.numbers || {}, null, 2);

  if (teleBullets) {
    teleBullets.innerHTML = "";
    const arr = Array.isArray(c.bullets) ? c.bullets : [];
    for (const b of arr.slice(0, 6)) {
      const li = document.createElement("li");
      li.textContent = String(b);
      teleBullets.appendChild(li);
    }
  }

  if (teleActions) {
    teleActions.innerHTML = "";
    const acts = Array.isArray(c.next_actions) ? c.next_actions : [];
    for (const a of acts.slice(0, 4)) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.textContent = a;
      chip.className = "chip"; // style in your CSS
      chip.addEventListener("click", async () => {
        const modality = modalitySel?.value || "language";
        try {
          const out = await steer(modality, a);
          renderSteerPayload(out);
        } catch (e) {
          if (steerOut) steerOut.textContent = `error: ${e.message}`;
        }
      });
      teleActions.appendChild(chip);
    }
  }

  setTeleStatus(Date.now() / 1000, c.llm_used, c.confidence);
}

function renderTeleState(s) {
  // s = { telemetry: {...}, ts: <unix seconds> }
  if (!s || typeof s !== "object") return;
  const tele = s.telemetry || {};
  if (teleNumbers) {
    // If counsel numbers will overwrite later, this still provides a raw view first.
    teleNumbers.textContent = JSON.stringify(tele, null, 2);
  }
  setTeleStatus(s.ts, undefined, undefined);
}

async function refreshTelemetry() {
  if (_teleBusy) return;
  _teleBusy = true;
  try {
    const [stateRes, counselRes] = await Promise.allSettled([
      getJSON("/telemetry/state"),
      getJSON("/telemetry/counsel"),
    ]);

    if (stateRes.status === "fulfilled") {
      renderTeleState(stateRes.value);
    }
    if (counselRes.status === "fulfilled") {
      renderCounsel(counselRes.value);
    }
  } catch (_) {
    // swallow; page shows last values
  } finally {
    _teleBusy = false;
  }
}

function startTelemetry() {
  if (_teleTimer) return;
  const ms = Math.max(300, parseInt((teleEveryMs?.value || "1200"), 10) || 1200);
  _teleTimer = setInterval(refreshTelemetry, ms);
  refreshTelemetry();
}

function stopTelemetry() {
  if (_teleTimer) {
    clearInterval(_teleTimer);
    _teleTimer = null;
  }
}

// ──────────────────────────────────────────────────────────────────────────
/** Populate adapters into the modality select (optional). */
async function refreshAdapters() {
  if (!modalitySel) return;
  try {
    const body = await getJSON("/adapters");
    const names = Array.isArray(body?.adapters) ? body.adapters : [];
    if (!names.length) return;
    const current = new Set([...modalitySel.querySelectorAll("option")].map(o => o.value));
    for (const n of names) {
      if (current.has(n)) continue;
      const opt = document.createElement("option");
      opt.value = n;
      opt.textContent = n;
      modalitySel.appendChild(opt);
    }
  } catch (_) {}
}

// ──────────────────────────────────────────────────────────────────────────
// Wire up events
// ──────────────────────────────────────────────────────────────────────────
on(btnHealth, "click", onHealth);

on(btnInfer , "click", onInfer);
on(inferText, "keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    onInfer();
  }
});

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

// Telemetry panel controls (optional)
on(btnTeleStart, "click", startTelemetry);
on(btnTeleStop , "click", stopTelemetry);
on(teleEveryMs , "change", () => {
  if (_teleTimer) { stopTelemetry(); startTelemetry(); }
});

// Do an initial health ping on load, populate adapters, and start telemetry if a container exists.
onHealth();
refreshAdapters();
if (teleCaption || teleBullets || teleNumbers || teleActions) {
  startTelemetry();
}
