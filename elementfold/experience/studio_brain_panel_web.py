# ElementFold · experience/studio_brain_panel_web.py
# ============================================================
# Web Dashboard for Local Brain / Telemetry Status
# ------------------------------------------------------------
# Now includes Start / Stop buttons for the brain auto-loop.
# ============================================================

HTML = """<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<title>ElementFold · Local Brain Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; margin: 2rem; background: #f8f8f9; color: #222; }
  h1 { font-weight: 600; }
  button {
    padding: .4rem .8rem; margin-right: .5rem;
    border: 1px solid #8883; border-radius: .4rem;
    background: #eee; cursor: pointer; font-size: .9rem;
  }
  button:hover { background: #ddd; }
  .status, .tele { background:#fff; border-radius:10px; padding:1rem 1.2rem;
                   margin-top:1rem; box-shadow:0 1px 4px #0001; }
  .bar { height:12px; border-radius:6px; background:#ddd; overflow:hidden; margin-top:0.4rem; }
  .fill { height:100%; background:linear-gradient(90deg,#2b8a3e,#94d82d); transition:width .3s; }
  .fill.warn { background:linear-gradient(90deg,#f59f00,#ffe066); }
  .fill.bad { background:linear-gradient(90deg,#e03131,#ffa8a8); }
  .mono { font-family: ui-monospace,monospace; font-size:0.9em; opacity:0.8; }
</style>
<body>
  <h1>🧠 ElementFold • Local Brain Dashboard</h1>
  <div>
    <button onclick="startLoop()">▶ Start Loop</button>
    <button onclick="stopLoop()">⏸ Stop Loop</button>
    <button onclick="fetchStatus()">⟳ Refresh</button>
  </div>

  <div id="loop" class="status">Loading...</div>
  <div id="tele" class="tele"></div>

  <script>
    async function startLoop(){
      await fetch('/brain/loop/start',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
      setTimeout(fetchStatus,1000);
    }
    async function stopLoop(){
      await fetch('/brain/loop/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
      setTimeout(fetchStatus,500);
    }
    async function fetchStatus(){
      try {
        const res = await fetch('/brain/loop/status');
        const data = await res.json();
        render(data);
      } catch(e) {
        document.getElementById('loop').innerText = "Error fetching: " + e;
      }
    }

    function bar(val,max=1,quality="good"){
      const pct = Math.min(100,Math.max(0,100*val/max));
      const cls = quality=="warn"?"fill warn":(quality=="bad"?"fill bad":"fill");
      return `<div class='bar'><div class='${cls}' style='width:${pct}%'></div></div>`;
    }

    function render(d){
      const loop=document.getElementById('loop');
      const tele=document.getElementById('tele');
      const st=d.status||"unknown";
      const dt=d.seconds_since_last_step ? `${d.seconds_since_last_step.toFixed(1)} s ago` : "";
      const dec=d.last_decision||{};
      const act=dec.action||"—";
      const comment=dec.comment||"";
      const params=dec.params?Object.entries(dec.params)
        .map(([k,v])=>`<span>${k}=${v}</span>`).join(", "):"";
      loop.innerHTML=`
        <h2>Status: ${st.toUpperCase()}</h2>
        <p><b>Last Action:</b> ${act}<br>
        <b>Params:</b> ${params}<br>
        <b>Comment:</b> ${comment}<br>
        <span class='mono'>Last step: ${dt}</span></p>`;
      const t=d.telemetry||{};
      tele.innerHTML=`
        <h3>Telemetry</h3>
        <p><b>κ</b> coherence ${bar(t.kappa||0,1,"good")}
        <b>p½</b> barrier ${bar(t.p_half||0,1,"warn")}
        <b>Variance</b> ${bar(t.resid_std||0,0.1,"bad")}
        <b>Margin Mean</b> ${bar(t.margin_mean||0,1,"good")}
        <b>Margin Min</b> ${bar(t.margin_min||0,1,"warn")}
        <span class='mono'>δ⋆=${(t.delta||0).toFixed(6)}</span></p>`;
    }

    fetchStatus();
    setInterval(fetchStatus, 3000);
  </script>
</body>
</html>
"""
