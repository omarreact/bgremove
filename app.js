// --- APP STATE ---
const state = {
  originalImg: null,
  maskCanvas: null,
  bgType: "color",
  bgColor: "transparent",
  bgImg: null,
  tool: "erase",
  brushSize: 20,
  isProcessing: false,
  isDrawing: false,
  viewMode: "edited", // NEW: 'edited' or 'original'
};

// --- DOM ELEMENTS ---
const els = {
  uploadView: document.getElementById("uploadView"),
  editorView: document.getElementById("editorView"),
  mainCanvas: document.getElementById("mainCanvas"),
  wrapper: document.getElementById("canvasWrapper"),
  loader: document.getElementById("loader"),
  progressFill: document.getElementById("progressFill"),
  loaderText: document.getElementById("loaderText"),
  downloadBtn: document.getElementById("downloadBtn"),
  newImageBtn: document.getElementById("newImageBtn"), // NEW
  brushCursor: document.getElementById("brushCursor"),
  toolErase: document.getElementById("toolErase"),
  toolRestore: document.getElementById("toolRestore"),
};

const ctx = els.mainCanvas.getContext("2d");
let session = null; // ONNX session

// --- INITIALIZATION ---

// File Inputs
document
  .getElementById("fileInput")
  .addEventListener("change", (e) => handleUpload(e.target.files[0]));
document
  .getElementById("bgFileInput")
  .addEventListener("change", (e) => handleBgUpload(e.target.files[0]));

// NEW: Reset / Upload New Button
els.newImageBtn.addEventListener("click", resetApp);

// NEW: View Toggle (Radio Buttons)
document.querySelectorAll('input[name="viewMode"]').forEach((radio) => {
  radio.addEventListener("change", (e) => {
    state.viewMode = e.target.id === "viewOriginal" ? "original" : "edited";
    render();
  });
});

// Color Picker
document
  .getElementById("colorPicker")
  .addEventListener("input", (e) => setBgColor(e.target.value));

// Brush Size
document.getElementById("brushSize").addEventListener("input", (e) => {
  state.brushSize = parseInt(e.target.value);
  updateCursor(e);
});

// Canvas Interaction for Painting
els.wrapper.addEventListener("mousedown", startPaint);
els.wrapper.addEventListener("mousemove", (e) => {
  movePaint(e);
  updateCursor(e);
});
els.wrapper.addEventListener("mouseup", endPaint);
els.wrapper.addEventListener("mouseleave", () => {
  endPaint();
  els.brushCursor.style.display = "none";
});
els.wrapper.addEventListener(
  "mouseenter",
  () => (els.brushCursor.style.display = "block"),
);

// Drag and Drop
const dropZone = document.getElementById("dropZone");
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-active");
});
dropZone.addEventListener("dragleave", (e) =>
  dropZone.classList.remove("drag-active"),
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-active");
  handleUpload(e.dataTransfer.files[0]);
});

// --- CORE LOGIC ---

async function handleUpload(file) {
  if (!file || !file.type.startsWith("image")) return;

  // Switch View
  els.uploadView.style.display = "none";
  els.editorView.style.display = "flex";
  els.newImageBtn.style.display = "block"; // Show "Upload New" button
  showLoader(true, "Loading Image...");

  const img = new Image();
  img.onload = async () => {
    state.originalImg = img;
    // Resize canvas to match image
    els.mainCanvas.width = img.width;
    els.mainCanvas.height = img.height;

    // Initialize Mask (Full White = Opaque for now)
    state.maskCanvas = document.createElement("canvas");
    state.maskCanvas.width = img.width;
    state.maskCanvas.height = img.height;

    // Run AI
    await runAI();
  };
  img.src = URL.createObjectURL(file);
}

function resetApp() {
  // Reset View
  els.editorView.style.display = "none";
  els.uploadView.style.display = "block";
  els.newImageBtn.style.display = "none";
  els.downloadBtn.disabled = true;

  // Reset Inputs
  document.getElementById("fileInput").value = "";
  document.getElementById("bgFileInput").value = "";

  // Reset State
  state.originalImg = null;
  state.maskCanvas = null;
  state.bgImg = null;
  state.bgType = "color";
  state.bgColor = "transparent";
  state.viewMode = "edited";

  // Reset Toggle UI
  document.getElementById("viewEdited").checked = true;
  document
    .querySelectorAll(".swatch")
    .forEach((s) => s.classList.remove("active"));
  document.querySelector(".swatch-transparent").classList.add("active");
}

async function runAI() {
  try {
    if (!session) {
      showLoader(true, "Loading AI Model...");
      await new Promise((r) => setTimeout(r, 100));
      session = await ort.InferenceSession.create("u2netp.onnx", {
        executionProviders: ["webgl", "wasm"],
      });
    }

    showLoader(true, "Removing Background...");
    await new Promise((r) => setTimeout(r, 50));

    const size = 320;
    const input = preprocess(state.originalImg, size);

    const feeds = {};
    feeds[session.inputNames[0]] = input;
    const output = await session.run(feeds);

    const outName = session.outputNames[0];
    const outputTensor = output[outName];

    processMask(outputTensor.data, size);

    showLoader(false);
    render();
    els.downloadBtn.disabled = false;
  } catch (e) {
    console.error(e);
    alert(
      "Error: " + e.message + "\nEnsure u2netp.onnx is in the same folder.",
    );
    resetApp();
  }
}

// --- RENDERING PIPELINE ---
function render() {
  if (!state.originalImg) return;

  const w = els.mainCanvas.width;
  const h = els.mainCanvas.height;
  ctx.clearRect(0, 0, w, h);

  // NEW: Check View Mode
  if (state.viewMode === "original") {
    // Just draw the original image and return
    ctx.drawImage(state.originalImg, 0, 0);
    return;
  }

  // --- Draw Background ---
  ctx.globalCompositeOperation = "source-over";
  if (state.bgType === "color") {
    if (state.bgColor !== "transparent") {
      ctx.fillStyle = state.bgColor;
      ctx.fillRect(0, 0, w, h);
    }
  } else if (state.bgType === "image" && state.bgImg) {
    drawCover(ctx, state.bgImg, w, h);
  }

  // --- Draw Masked Foreground ---
  const tempC = document.createElement("canvas");
  tempC.width = w;
  tempC.height = h;
  const tCtx = tempC.getContext("2d");

  tCtx.drawImage(state.originalImg, 0, 0);
  tCtx.globalCompositeOperation = "destination-in";
  tCtx.drawImage(state.maskCanvas, 0, 0);

  ctx.drawImage(tempC, 0, 0);

  // Update Download Link
  els.downloadBtn.onclick = () => {
    const link = document.createElement("a");
    link.download = "edited-image.png";
    link.href = els.mainCanvas.toDataURL("image/png");
    link.click();
  };
}

// --- MANUAL BRUSHING ---
function startPaint(e) {
  // Disable painting if in "Original" view mode
  if (state.viewMode === "original") return;

  state.isDrawing = true;
  paint(e);
}
function endPaint() {
  state.isDrawing = false;
}
function movePaint(e) {
  if (state.isDrawing) paint(e);
}

function paint(e) {
  const rect = els.mainCanvas.getBoundingClientRect();
  const scaleX = els.mainCanvas.width / rect.width;
  const scaleY = els.mainCanvas.height / rect.height;

  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  const mCtx = state.maskCanvas.getContext("2d");
  mCtx.beginPath();
  mCtx.arc(x, y, state.brushSize / 2, 0, Math.PI * 2);

  mCtx.globalCompositeOperation = "source-over";

  if (state.tool === "erase") {
    mCtx.globalCompositeOperation = "destination-out";
    mCtx.fillStyle = "rgba(0,0,0,1)";
    mCtx.fill();
  } else {
    mCtx.globalCompositeOperation = "source-over";
    mCtx.fillStyle = "rgba(255,255,255,1)";
    mCtx.fill();
  }

  render();
}

// --- TOOLS & HELPERS ---

window.setTool = function (tool) {
  state.tool = tool;
  els.toolErase.classList.toggle("active", tool === "erase");
  els.toolRestore.classList.toggle("active", tool === "restore");
};

window.setBgColor = function (color, el) {
  state.bgType = "color";
  state.bgColor = color;
  if (el) {
    document
      .querySelectorAll(".swatch")
      .forEach((s) => s.classList.remove("active"));
    el.classList.add("active");
  }
  // Force view to Edited if changing background
  if (state.viewMode === "original") {
    state.viewMode = "edited";
    document.getElementById("viewEdited").checked = true;
  }
  render();
};

function handleBgUpload(file) {
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    state.bgType = "image";
    state.bgImg = img;
    document
      .querySelectorAll(".swatch")
      .forEach((s) => s.classList.remove("active"));

    // Force view to Edited
    if (state.viewMode === "original") {
      state.viewMode = "edited";
      document.getElementById("viewEdited").checked = true;
    }
    render();
  };
  img.src = URL.createObjectURL(file);
}

function updateCursor(e) {
  // Hide cursor if in Original mode
  if (state.viewMode === "original") {
    els.brushCursor.style.display = "none";
    return;
  }

  const brush = els.brushCursor;
  brush.style.width = state.brushSize + "px";
  brush.style.height = state.brushSize + "px";
  brush.style.left = e.clientX + "px";
  brush.style.top = e.clientY + "px";
  brush.style.display = "block";
}

function showLoader(show, text) {
  els.loader.style.display = show ? "flex" : "none";
  if (text) els.loaderText.textContent = text;
  if (show) {
    els.progressFill.style.width = "100%";
  } else {
    els.progressFill.style.width = "0%";
  }
}

// --- AI UTILS (Standard U2Net Pre/Post Processing) ---
function preprocess(image, size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  const r = Math.min(size / image.width, size / image.height);
  const w = Math.round(image.width * r);
  const h = Math.round(image.height * r);
  ctx.drawImage(image, (size - w) / 2, (size - h) / 2, w, h);

  const imgData = ctx.getImageData(0, 0, size, size).data;
  const floatData = new Float32Array(3 * size * size);

  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < size * size; i++) {
      const v = imgData[i * 4 + c] / 255.0;
      const mean = [0.485, 0.456, 0.406][c];
      const std = [0.229, 0.224, 0.225][c];
      floatData[i + c * size * size] = (v - mean) / std;
    }
  }
  return new ort.Tensor("float32", floatData, [1, 3, size, size]);
}

function processMask(data, size) {
  const tCanvas = document.createElement("canvas");
  tCanvas.width = size;
  tCanvas.height = size;
  const tCtx = tCanvas.getContext("2d");
  const imgData = tCtx.createImageData(size, size);

  let min = Infinity,
    max = -Infinity;
  for (let v of data) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;

  for (let i = 0; i < data.length; i++) {
    const val = ((data[i] - min) / range) * 255;
    imgData.data[i * 4] = val;
    imgData.data[i * 4 + 1] = val;
    imgData.data[i * 4 + 2] = val;
    imgData.data[i * 4 + 3] = 255;
  }
  tCtx.putImageData(imgData, 0, 0);

  const mCtx = state.maskCanvas.getContext("2d");
  mCtx.drawImage(
    tCanvas,
    0,
    0,
    state.maskCanvas.width,
    state.maskCanvas.height,
  );

  const raw = mCtx.getImageData(
    0,
    0,
    state.maskCanvas.width,
    state.maskCanvas.height,
  );
  for (let i = 0; i < raw.data.length; i += 4) {
    const v = raw.data[i];
    raw.data[i + 3] = v;
    raw.data[i] = 255;
    raw.data[i + 1] = 255;
    raw.data[i + 2] = 255;
  }
  mCtx.putImageData(raw, 0, 0);
}

function drawCover(ctx, img, w, h) {
  const r = Math.max(w / img.width, h / img.height);
  const nw = img.width * r,
    nh = img.height * r;
  const cx = (w - nw) / 2,
    cy = (h - nh) / 2;
  ctx.drawImage(img, cx, cy, nw, nh);
}
