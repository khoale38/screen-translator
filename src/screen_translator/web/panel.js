const $ = (id) => document.getElementById(id);
let state = null;
let pendingRegion = null;
const lines = new Map();

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function showError(message) {
  $("error").hidden = !message;
  $("error").textContent = message || "";
}

function fillSelect(select, entries, value) {
  select.replaceChildren(...entries.map(([v, label]) => new Option(label, v)));
  select.value = value;
}

function regionText(region) {
  if (!region) return "Whole screen";
  const pct = (v) => `${Math.round(v * 100)}%`;
  return `Area ${pct(region.width)} × ${pct(region.height)} at ${pct(region.left)}, ${pct(region.top)}`;
}

function setStatus(status, running) {
  const label = status.state === "running" && !running ? "stopped" : status.state;
  $("status").textContent = label;
  $("status").className = `pill ${label}`;
  $("toggle").textContent = running || label === "loading" ? "Stop" : "Start";
  showError(status.error);
}

function renderModels() {
  const s = state.settings;
  fillSelect($("model"), state.models.map((m) => [m.key, `${m.label}${m.downloaded ? " ✓" : ""}`]), s.model);
  const model = state.models.find((m) => m.key === s.model);
  const dl = state.download;
  const active = dl && dl.active;
  $("download-row").hidden = !!(model && model.downloaded && !active) || !!s.custom_model_path;
  $("download").textContent = active ? "Cancel" : "Download";
  $("progress").hidden = !active;
  if (active) {
    $("progress").value = dl.done / dl.total;
    $("progress-label").textContent = `${(dl.done / 1e9).toFixed(2)} / ${(dl.total / 1e9).toFixed(2)} GB`;
  } else {
    $("progress-label").textContent = dl && dl.error ? dl.error : "";
  }
}

function render() {
  const s = state.settings;
  const langs = Object.entries(state.languages);
  fillSelect($("source_lang"), langs, s.source_lang);
  fillSelect($("target_lang"), langs.filter(([k]) => k !== "auto"), s.target_lang);
  $("context_lines").value = s.context_lines;
  $("context_minutes").value = s.context_ttl_s / 60;
  $("notes").value = s.notes;
  $("glossary").value = Object.entries(s.glossary).map(([a, b]) => `${a} = ${b}`).join("\n");
  $("custom_model_path").value = s.custom_model_path;
  $("region-label").textContent = regionText(s.region);
  $("ocr-hint").textContent = state.ocr_hint;
  $("ocr-hint").hidden = !state.ocr_hint;
  $("overlay-url").value = `http://127.0.0.1:${s.port}/overlay`;
  renderModels();
  setStatus(state.status, state.running);
  state.lines.forEach(onLine);
}

async function save(patch) {
  try {
    state = { ...state, ...(await api("/api/settings", patch)) };
    renderModels();
    $("region-label").textContent = regionText(state.settings.region);
  } catch (err) {
    showError(err.message);
  }
}

function onLine(event) {
  lines.set(event.id, event);
  while (lines.size > 20) lines.delete(lines.keys().next().value);
  const items = [...lines.values()].reverse().map((l) => {
    const el = document.createElement("div");
    el.className = "item" + (l.final ? "" : " pending");
    const src = Object.assign(document.createElement("div"), { className: "src", textContent: l.source });
    const dst = Object.assign(document.createElement("div"), { className: "dst", textContent: l.translation || "…" });
    el.append(src, dst);
    if (l.final && l.ms !== undefined) {
      el.append(Object.assign(document.createElement("div"), {
        className: "meta", textContent: `${l.ms} ms · ${l.context_lines} lines of context`,
      }));
    }
    return el;
  });
  $("live").replaceChildren(...items);
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    if (event.type === "line") onLine(event);
    else if (event.type === "ocr") $("ocr").textContent = event.text ? `${event.text} (${event.ms} ms)` : "no text";
    else if (event.type === "status") {
      state.status = event;
      state.running = event.state === "running" || event.state === "loading";
      setStatus(event, state.running);
    } else if (event.type === "download") {
      state.download = event;
      if (event.finished) {
        api("/api/state").then((s) => { state = s; renderModels(); });
      } else renderModels();
    }
  };
  ws.onclose = () => setTimeout(connect, 1500);
}

$("toggle").onclick = async () => {
  try {
    state = await api(state.running || state.status.state === "loading" ? "/api/stop" : "/api/start", {});
    setStatus(state.status, state.running);
  } catch (err) {
    showError(err.message);
  }
};

for (const id of ["source_lang", "target_lang", "monitor"]) {
  $(id).onchange = () => save({ [id]: id === "monitor" ? Number($(id).value) : $(id).value });
}
$("model").onchange = () => save({ model: $("model").value });
$("custom_model_path").onchange = () => save({ custom_model_path: $("custom_model_path").value.trim() });

$("save-context").onclick = () => {
  const glossary = {};
  for (const line of $("glossary").value.split("\n")) {
    const [src, ...rest] = line.split("=");
    if (src.trim() && rest.length) glossary[src.trim()] = rest.join("=").trim();
  }
  save({
    context_lines: Number($("context_lines").value),
    context_ttl_s: Number($("context_minutes").value) * 60,
    notes: $("notes").value,
    glossary,
  });
};
$("clear-context").onclick = () => api("/api/context/clear", {});

$("download").onclick = () => {
  const active = state.download && state.download.active;
  (active ? api("/api/models/download/cancel", {}) : api(`/api/models/${state.settings.model}/download`, {}))
    .catch((err) => showError(err.message));
};

$("copy").onclick = () => navigator.clipboard.writeText($("overlay-url").value);
$("full").onclick = () => save({ region: null });
$("test-ocr").onclick = async () => {
  $("ocr-test").textContent = "Reading…";
  try {
    const result = await api("/api/ocr/test", {});
    $("ocr-test").textContent = `${result.engine}: ${result.text || "(no text found)"}`;
  } catch (err) {
    $("ocr-test").textContent = err.message;
  }
};

$("pick").onclick = () => {
  $("picker").hidden = false;
  $("sel").style.display = "none";
  pendingRegion = null;
  $("shot").src = `/api/screenshot?monitor=${$("monitor").value}&t=${Date.now()}`;
};
$("shot").onerror = async () => {
  $("picker").hidden = true;
  const res = await fetch($("shot").src).catch(() => null);
  showError(res ? await res.text() : "Could not capture the screen.");
};
$("cancel-pick").onclick = () => { $("picker").hidden = true; };
$("save-region").onclick = () => {
  if (pendingRegion) save({ region: pendingRegion });
  $("picker").hidden = true;
};

(function regionDrag() {
  const wrap = $("shot-wrap");
  const sel = $("sel");
  let start = null;
  const point = (e) => {
    const r = $("shot").getBoundingClientRect();
    return {
      x: Math.min(Math.max(e.clientX - r.left, 0), r.width),
      y: Math.min(Math.max(e.clientY - r.top, 0), r.height),
      w: r.width, h: r.height,
    };
  };
  wrap.onpointerdown = (e) => {
    start = point(e);
    wrap.setPointerCapture(e.pointerId);
    e.preventDefault();
  };
  wrap.onpointermove = (e) => {
    if (!start) return;
    const p = point(e);
    const left = Math.min(start.x, p.x), top = Math.min(start.y, p.y);
    const width = Math.abs(p.x - start.x), height = Math.abs(p.y - start.y);
    Object.assign(sel.style, {
      display: "block", left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px`,
    });
    if (width > 4 && height > 4) {
      pendingRegion = { left: left / p.w, top: top / p.h, width: width / p.w, height: height / p.h };
    }
  };
  wrap.onpointerup = () => { start = null; };
})();

async function init() {
  state = await api("/api/state");
  const monitors = await api("/api/monitors").catch(() => []);
  fillSelect($("monitor"), monitors.map((m) => [String(m.index), `Screen ${m.index} (${m.width}×${m.height})`]),
    String(state.settings.monitor));
  render();
  connect();
}

init().catch((err) => showError(err.message));
