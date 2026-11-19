const params = new URLSearchParams(location.search);
const maxLines = Math.max(1, Number(params.get("lines") || 2));
const showSource = params.get("source") === "1";
const holdSeconds = Number(params.get("hold") || 0);
const root = document.documentElement.style;
if (params.get("size")) root.setProperty("--size", `${Number(params.get("size"))}px`);
if (params.get("bg")) root.setProperty("--bg", `rgba(0,0,0,${Number(params.get("bg"))})`);
if (params.get("color")) root.setProperty("--fg", `#${params.get("color").replace(/[^0-9a-f]/gi, "")}`);
if (params.get("pagebg")) document.body.style.background = `#${params.get("pagebg").replace(/[^0-9a-f]/gi, "")}`;
if (params.get("position") === "top") document.body.classList.add("top");
if (params.get("align") === "left") document.body.classList.add("left");

const container = document.getElementById("lines");
const lines = [];
let hideTimer = null;

function render() {
  container.replaceChildren(...lines.map((line, i) => {
    const el = document.createElement("div");
    el.className = "line" + (i < lines.length - 1 ? " old" : "") + (line.final ? "" : " pending");
    if (showSource) {
      const src = document.createElement("span");
      src.className = "src";
      src.textContent = line.source;
      el.append(src);
    }
    el.append(document.createTextNode(line.translation || "…"));
    return el;
  }));
}

function scheduleHide() {
  clearTimeout(hideTimer);
  if (holdSeconds > 0) hideTimer = setTimeout(clear, holdSeconds * 1000);
}

function clear() {
  lines.length = 0;
  render();
}

function onLine(event) {
  const existing = lines.find((l) => l.id === event.id);
  if (existing) Object.assign(existing, event);
  else lines.push(event);
  while (lines.length > maxLines) lines.shift();
  render();
  if (event.final) scheduleHide();
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    if (event.type === "line") onLine(event);
    else if (event.type === "clear") clear();
  };
  ws.onclose = () => setTimeout(connect, 1500);
}

connect();
