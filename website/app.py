"""Backend for the project website: serves the page and analyses uploaded images.

An upload is checked in two ways: its provenance metadata (signed Content Credentials, AI labels) and its
pixels (CLIP features of a tile grid and a linear classifier). Valid signed credentials decide the verdict;
otherwise the pixel detector does.
"""

import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pillow_heif
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from transformers import CLIPVisionModelWithProjection

import provenance

pillow_heif.register_heif_opener()
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", os.cpu_count() or 2)))

HERE = Path(__file__).parent
MODEL_REPO = os.environ.get("MODEL_REPO", "tanish-237/ai-image-forensics-models")
MODEL_DIR = os.environ.get("MODEL_DIR")
# Tiles per image. Tested on 206 images (our own AI images, phone photos from 14 unseen models):
# 9 tiles 89.9% balanced accuracy, 16 tiles 94.3%, 25 tiles 94.6%; 4 tiles was worse and 36 gave no gain.
TILE_CHOICES = (9, 16, 25)
# The E11 family guess (ChatGPT, Gemini, Grok, ...) named the right app for only 5 of our 62 own images,
# so it is off unless SHOW_FAMILIES=1; signed Content Credentials still name the tool when present.
SHOW_FAMILIES = os.environ.get("SHOW_FAMILIES") == "1"
MAX_TILES = int(os.environ.get("MAX_TILES", "16"))
MAX_UPLOAD_MB = 25
SIZE = 224

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class Detector:
    """Frozen CLIP ViT-L/14 features of each tile and a logistic-regression classifier."""

    def __init__(self, folder):
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text())
        # E9 deployment detector if present, otherwise the E8 detector of notebook 18
        chosen = config.get("detector") or {"file": "m2_lr.npz", "threshold": config["thresholds"]["M2"],
                                            "name": config["names"]["M2"]}
        self.name, self.threshold = chosen["name"], chosen["threshold"]
        self.clip = CLIPVisionModelWithProjection.from_pretrained(config["clip_name"]).eval()
        self.clip_size = self.clip.config.image_size
        lr = np.load(folder / chosen["file"])
        self.w, self.b = torch.from_numpy(lr["coef"]).float().view(-1), float(lr["intercept"])
        # E11: which generator family (ChatGPT, Gemini, Grok, ...), if the model repository has it
        self.families = None
        if "attributor" in config:
            attr = np.load(folder / config["attributor"]["file"])
            self.families = config["attributor"]["classes"]
            self.attr_w, self.attr_b = torch.from_numpy(attr["coef"]).float(), torch.from_numpy(attr["intercept"]).float()

    @torch.no_grad()
    def embed(self, tiles):
        x = torch.from_numpy(tiles).permute(0, 3, 1, 2).float() / 255
        if self.clip_size != SIZE:
            x = F.interpolate(x, size=self.clip_size, mode="bicubic", align_corners=False)
        return F.normalize(self.clip(pixel_values=(x - CLIP_MEAN) / CLIP_STD).image_embeds, dim=1)

    def score(self, emb):
        return torch.sigmoid(emb @ self.w + self.b).numpy()

    def attribute(self, emb):
        """Average family probabilities over the tiles; returns the AI families, most likely first."""
        if self.families is None:
            return None
        p = torch.softmax(emb @ self.attr_w.T + self.attr_b, dim=1).mean(dim=0).numpy()
        ai = {f: float(q) for f, q in zip(self.families, p) if f != "Real"}
        total = sum(ai.values())
        return [{"family": f, "probability": q / total} for f, q in sorted(ai.items(), key=lambda kv: -kv[1])]


def calibrate(p, threshold):
    """Rescale a score so that the calibrated threshold maps to 0.5 (piecewise linear, monotonic)."""
    p = np.asarray(p, dtype=np.float64)
    return np.where(p < threshold, 0.5 * p / threshold, 0.5 + 0.5 * (p - threshold) / (1 - threshold))


def tile_positions(width, height, max_tiles=MAX_TILES):
    per_side = int(np.sqrt(max_tiles))
    xs = np.unique(np.linspace(0, width - SIZE, min(per_side, max(1, width // SIZE))).astype(int) // 16 * 16)
    ys = np.unique(np.linspace(0, height - SIZE, min(per_side, max(1, height // SIZE))).astype(int) // 16 * 16)
    return [(int(x), int(y)) for y in ys for x in xs]


def model_folder():
    if MODEL_DIR:
        return MODEL_DIR
    try:
        return snapshot_download(MODEL_REPO, token=os.environ.get("HF_TOKEN"))
    except Exception:
        # no internet: use the copy downloaded on an earlier run
        os.environ["HF_HUB_OFFLINE"] = "1"
        return snapshot_download(MODEL_REPO, local_files_only=True)


detector = Detector(model_folder())
app = FastAPI(title="AI Image Forensics")
# the static copy of the page (hosted elsewhere) calls this API from another address
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), tiles: int = Form(MAX_TILES)):
    start = time.time()
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Please upload an image smaller than {MAX_UPLOAD_MB} MB.")
    try:
        original = Image.open(io.BytesIO(data))
        origin = provenance.check(data, original)
        img = ImageOps.exif_transpose(original).convert("RGB")
    except Exception:
        raise HTTPException(400, "This file could not be read as an image.")

    notes = []
    width, height = img.size
    if min(img.size) < SIZE:
        factor = SIZE / min(img.size)
        img = img.resize((max(SIZE, round(width * factor)), max(SIZE, round(height * factor))), Image.BICUBIC)
        notes.append("The image is smaller than 224 pixels and was enlarged, which makes the pixel analysis less reliable.")

    if tiles not in TILE_CHOICES:
        raise HTTPException(400, f"tiles must be one of {TILE_CHOICES}.")
    positions = tile_positions(*img.size, max_tiles=tiles)
    tiles = np.stack([np.asarray(img.crop((x, y, x + SIZE, y + SIZE))) for x, y in positions])
    emb = detector.embed(tiles)
    tile_scores = detector.score(emb)
    pixel_score = float(calibrate(tile_scores.mean(), detector.threshold))

    if origin["decision"] in ("ai", "ai_label"):
        verdict, decided_by = "AI-generated", "credentials" if origin["decision"] == "ai" else "label"
    elif origin["decision"] == "camera":
        verdict, decided_by = "Real photograph", "credentials"
    else:
        verdict, decided_by = ("AI-generated" if pixel_score >= 0.5 else "Real photograph"), "pixels"
    if origin["credentials"] and origin["credentials"]["state"] == "Invalid":
        notes.append("The file carries Content Credentials, but they do not match the image (it was changed after "
                     "signing, or the signature is broken), so they were not used.")

    return {
        "verdict": verdict,
        "decided_by": decided_by,
        "provenance": origin,
        "pixels": {"name": detector.name, "probability_ai": pixel_score},
        "families": detector.attribute(emb) if SHOW_FAMILIES and verdict == "AI-generated" else None,
        "tiles": [{"x": x, "y": y, "size": SIZE, "probability_ai": float(calibrate(p, detector.threshold))}
                  for (x, y), p in zip(positions, tile_scores)],
        "analysed_size": list(img.size),
        "original_size": [width, height],
        "notes": notes,
        "seconds": round(time.time() - start, 2),
    }


@app.get("/api/info")
def info():
    return {"model": detector.name, "max_tiles": MAX_TILES, "tile_choices": list(TILE_CHOICES)}


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")
