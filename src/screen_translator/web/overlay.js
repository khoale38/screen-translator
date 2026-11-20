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

const showScreen = params.get("screen") !== "0";
const screenLayer = document.getElementById("screen");
const container = document.getElementById("lines");
let blocks = [];
const screenTexts = new Map();
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

const sameText = (a, b) => a.replace(/\s+/g, " ").trim().toLowerCase() === b.replace(/\s+/g, " ").trim().toLowerCase();

function fit(el, size) {
  for (let px = size; px >= Math.max(8, size * 0.6); px -= 1) {
    el.style.fontSize = `${px}px`;
    if (el.scrollHeight <= el.clientHeight + 1 && el.scrollWidth <= el.clientWidth + 1) return;
  }
  el.style.height = "auto";
}

function renderScreen() {
  if (!showScreen) return;
  const width = window.innerWidth;
  const height = window.innerHeight;
  screenLayer.replaceChildren();
  for (const block of blocks) {
    const known = screenTexts.get(block.text);
    if (!known || (known.final && sameText(known.text, block.text))) continue;
    const [x, y, w, h] = block.box;
    const el = document.createElement("div");
    el.className = "patch" + (known.final ? "" : " pending");
    Object.assign(el.style, {
      left: `${x * width - 3}px`, top: `${y * height - 2}px`,
      width: `${w * width + 6}px`, height: `${h * height + 4}px`,
      background: block.bg, color: block.fg,
    });
    el.textContent = known.text;
    screenLayer.append(el);
    fit(el, Math.max(9, Math.round(block.line_height * height * 0.85)));
  }
}

function onScreen(event) {
  blocks = event.blocks;
  for (const block of blocks) {
    if (block.translation) screenTexts.set(block.text, { text: block.translation, final: true });
  }
  renderScreen();
}

function onScreenText(event) {
  screenTexts.set(event.source, { text: event.translation, final: event.final });
  if (blocks.some((b) => b.text === event.source)) renderScreen();
}

function clearScreen() {
  blocks = [];
  screenTexts.clear();
  renderScreen();
}

window.addEventListener("resize", renderScreen);

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    if (event.type === "line") onLine(event);
    else if (event.type === "screen") onScreen(event);
    else if (event.type === "screen_text") onScreenText(event);
    else if (event.type === "status" && event.state === "stopped") clearScreen();
    else if (event.type === "clear" && event.area === undefined) { clear(); clearScreen(); }
    else if (event.type === "clear") clear();
  };
  ws.onclose = () => setTimeout(connect, 1500);
}

connect();
