# ElementFold · experience/studio_brain_panel_web.py
# ============================================================
# Web Dashboard for Local Brain / Telemetry Status
# ------------------------------------------------------------
# A minimal dependency-free HTML+JS page that polls
# /brain/loop/status every few seconds and renders the state.
# ============================================================

from __future__ import annotations

HTML = """<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<title>ElementFold · Local Brain Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; margin: 2rem; background: #f7f7f8; color: #222; }
  h1 { font-weight: 600; }
  .status { margin-top: 1rem; }
  .box { background: #fff; border-radius: 10px; padding: 1.2rem 1.5rem; margin: 1rem 0; box-shadow: 0 2px 6px #0001; }
  .bar { height: 12px; border-radius: 6px; background: #ddd; overflow: hidden; margin-top: 0.4rem; }
  .fill { height: 100%; background: linear-gradient(90deg,#2b8a3e,#94d82d); transition: width 0.3s; }
  .fill.warn { background: linear-gradient(90deg,#f59f00,#ffe066); }
  .fill.bad { background: linear-gradient(90deg,#e03131,#ffa8a8); }
  .monosmall { font-family: ui-monospace,monospace; font-size: 0.9em; opacity:0.75; }
</style>
<body>
  <h1>🧠 ElementFold • Local Brain</h1>
  <div id="loop" class="box status">Loading...</div>
  <div id="tele" class="box"></div>
  <script>
    async function fetchStatus() {
      try {
        const res = await fetch("/brain/loop/status");
        const data = await res.json();
        render(data);
      } catch(e) {
        document.getElementById('loop').innerText = "Error fetching: " + e;
      }
    }

    function bar(val, max=1, quality="good") {
      const pct = Math.min(100, Math.max(0, 100*val/max));
      const cls = quality=="warn"?"fill warn":(quality=="bad"?"fill bad":"fill");
      return `<div class='bar'><div class='${cls}' style='width:${pct}%' title='${val.toFixed(3)}'></div></div>`;
    }

    function render(d) {
      const loop = document.getElementById('loop');
      const tele = document.getElementById('tele');
      const st = d.status || "unknown";
      const dt = d.seconds_since_last_step ? `${d.seconds_since_last_step.toFixed(1)} s ago` : "";
      const dec = d.last_decision || {};
      const act = dec.action || "—";
      const comment = dec.comment || "";
      const params = dec.params ? Object.entries(dec.params).map(([k,v]) => `<span>${k}=${v}</span>`).join(", ") : "";

      loop.innerHTML = `
        <h2>Loop: ${st.toUpperCase()}</h2>
        <p><b>Last Action:</b> ${act} <br>
           <b>Params:</b> ${params}<br>
           <b>Comment:</b> ${comment}<br>
           <span class='monosmall'>Last step: ${dt}</span></p>
      `;

      const t = d.telemetry || {};
      tele.innerHTML = `
        <h3>Telemetry</h3>
        <p><b>κ</b> coherence ${bar(t.kappa || 0, 1, "good")}
           <b>p½</b> barrier ${bar(t.p_half || 0, 1, "warn")}
           <b>Variance</b> ${bar(t.resid_std || 0, 0.1, "bad")}
           <b>Margin Mean</b> ${bar(t.margin_mean || 0, 1, "good")}
           <b>Margin Min</b> ${bar(t.margin_min || 0, 1, "warn")}
           <b>Phase Mean:</b> ${(t.phase_mean||0).toFixed(3)}<br>
           <span class='monosmall'>δ⋆ = ${(t.delta||0).toFixed(6)}</span>
        </p>`;
    }

    fetchStatus();
    setInterval(fetchStatus, 3000);
  </script>
</body>
</html>
"""
