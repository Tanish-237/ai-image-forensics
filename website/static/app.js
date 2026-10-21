const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const canvas = document.getElementById("preview");
const ctx = canvas.getContext("2d");

const views = {
  empty: document.getElementById("result-empty"),
  busy: document.getElementById("result-busy"),
  error: document.getElementById("result-error"),
  body: document.getElementById("result-body"),
};

function show(name) {
  for (const [key, el] of Object.entries(views)) el.hidden = key !== name;
}

function colourFor(p) {
  const style = getComputedStyle(document.documentElement);
  return p >= 0.5 ? style.getPropertyValue("--ai").trim() : style.getPropertyValue("--real").trim();
}

function percent(p) {
  return `${(p * 100).toFixed(1)}%`;
}

let currentImage = null;

function drawImage(img, tiles, analysedSize) {
  const maxWidth = drop.clientWidth - 32;
  const scale = Math.min(1, maxWidth / img.width, 520 / img.height);
  canvas.width = Math.round(img.width * scale);
  canvas.height = Math.round(img.height * scale);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  if (!tiles) return;

  // tiles are in the coordinates of the analysed image, which may have been enlarged
  const sx = canvas.width / analysedSize[0];
  const sy = canvas.height / analysedSize[1];
  for (const t of tiles) {
    const colour = colourFor(t.probability_ai);
    const strength = Math.abs(t.probability_ai - 0.5) * 2;
    ctx.fillStyle = colour;
    ctx.globalAlpha = 0.12 + 0.33 * strength;
    ctx.fillRect(t.x * sx, t.y * sy, t.size * sx, t.size * sy);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.strokeRect(t.x * sx + 1, t.y * sy + 1, t.size * sx - 2, t.size * sy - 2);
    const label = `${Math.round(t.probability_ai * 100)}`;
    ctx.font = "600 12px Inter, sans-serif";
    const w = ctx.measureText(label).width + 10;
    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fillRect(t.x * sx + 4, t.y * sy + 4, w, 18);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, t.x * sx + 9, t.y * sy + 17);
  }
}

function render(result) {
  document.getElementById("verdict").textContent = result.verdict;
  document.getElementById("verdict").style.color = colourFor(result.probability_ai);
  document.getElementById("meter-fill").style.left = `${result.probability_ai * 100}%`;
  document.getElementById("prob").textContent = `AI score ${percent(result.probability_ai)}`;

  const detectors = document.getElementById("detectors");
  detectors.innerHTML = "";
  for (const [name, p] of Object.entries(result.detectors)) {
    const li = document.createElement("li");
    li.innerHTML = `<span></span><span class="value">${percent(p)}</span><div class="track"><div class="fill"></div></div>`;
    li.querySelector("span").textContent = name;
    const fill = li.querySelector(".fill");
    fill.style.width = `${p * 100}%`;
    fill.style.background = colourFor(p);
    detectors.appendChild(li);
  }

  const sources = document.getElementById("sources");
  sources.innerHTML = "";
  for (const s of result.closest_sources) {
    const li = document.createElement("li");
    li.innerHTML = "<span></span><span></span>";
    li.children[0].textContent = s.source;
    li.children[1].textContent = percent(s.probability);
    sources.appendChild(li);
  }

  document.getElementById("notes").textContent = result.notes.join(" ");
  const [w, h] = result.original_size;
  document.getElementById("meta").textContent =
    `${w} × ${h} pixels · ${result.tiles.length} tiles analysed at full resolution · numbers on tiles are calibrated AI scores (%) · ${result.seconds} s`;
  if (currentImage) {
    drawImage(currentImage, result.tiles, result.analysed_size);
  } else {
    document.getElementById("preview-wrap").hidden = true;
    const empty = document.getElementById("drop-empty");
    empty.hidden = false;
    empty.innerHTML = "<strong>Analysed</strong><span class='muted'>This browser cannot preview this file format, so the tile map is not shown.</span>";
  }
  show("body");
}

async function analyse(file) {
  show("busy");
  document.getElementById("drop-empty").hidden = true;
  document.getElementById("preview-wrap").hidden = false;

  currentImage = null;
  try {
    currentImage = await createImageBitmap(file, { imageOrientation: "from-image" });
    drawImage(currentImage);
  } catch {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // e.g. HEIC in browsers that cannot display it
  }

  const form = new FormData();
  form.append("file", file);
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The image could not be analysed.");
    render(data);
  } catch (err) {
    views.error.textContent = err.message;
    show("error");
  }
}

fileInput.addEventListener("change", () => fileInput.files[0] && analyse(fileInput.files[0]));
for (const type of ["dragenter", "dragover"]) {
  drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add("over"); });
}
for (const type of ["dragleave", "drop"]) {
  drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.remove("over"); });
}
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && analyse(e.dataTransfer.files[0]));
