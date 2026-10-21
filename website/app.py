"""Backend for the project website: serves the page and analyses uploaded images."""

import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pillow_heif
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from torchvision.models import resnet50
from transformers import CLIPVisionModelWithProjection

pillow_heif.register_heif_opener()
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", os.cpu_count() or 2)))

HERE = Path(__file__).parent
MODEL_REPO = os.environ.get("MODEL_REPO", "tanish-237/ai-image-forensics-models")
MODEL_DIR = os.environ.get("MODEL_DIR")
MAX_TILES = int(os.environ.get("MAX_TILES", "9"))
MAX_UPLOAD_MB = 25
SIZE = 224

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_resnet(path, outputs):
    net = resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, outputs)
    net.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
    return net.eval()


class Models:
    def __init__(self, folder):
        folder = Path(folder)
        self.config = json.loads((folder / "config.json").read_text())
        self.m1 = load_resnet(folder / "m1.pt", 1)
        self.attributor = load_resnet(folder / "attributor.pt", len(self.config["sources"]))
        self.clip = CLIPVisionModelWithProjection.from_pretrained(self.config["clip_name"]).eval()
        self.clip_size = self.clip.config.image_size
        lr = np.load(folder / "m2_lr.npz")
        self.w, self.b = torch.from_numpy(lr["coef"]).float().view(-1), float(lr["intercept"])

    @torch.no_grad()
    def score_tiles(self, tiles):
        x = torch.from_numpy(tiles).permute(0, 3, 1, 2).float() / 255
        imagenet_x = (x - IMAGENET_MEAN) / IMAGENET_STD
        m1 = torch.sigmoid(self.m1(imagenet_x).squeeze(1))
        clip_x = x if self.clip_size == SIZE else F.interpolate(x, size=self.clip_size, mode="bicubic", align_corners=False)
        emb = F.normalize(self.clip(pixel_values=(clip_x - CLIP_MEAN) / CLIP_STD).image_embeds, dim=1)
        m2 = torch.sigmoid(emb @ self.w + self.b)
        return m1.numpy(), m2.numpy()

    @torch.no_grad()
    def attribute(self, tile):
        x = (torch.from_numpy(tile[None]).permute(0, 3, 1, 2).float() / 255 - IMAGENET_MEAN) / IMAGENET_STD
        probs = torch.softmax(self.attributor(x), dim=1)[0].numpy()
        ranked = sorted(zip(self.config["sources"], probs.tolist()), key=lambda s: -s[1])
        return [{"source": s, "probability": p} for s, p in ranked[:3]]


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


models = Models(model_folder())
app = FastAPI(title="AI Image Forensics")
# the static copy of the page (hosted elsewhere) calls this API from another address
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    start = time.time()
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Please upload an image smaller than {MAX_UPLOAD_MB} MB.")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(400, "This file could not be read as an image.")

    notes = []
    width, height = img.size
    if min(img.size) < SIZE:
        factor = SIZE / min(img.size)
        img = img.resize((max(SIZE, round(width * factor)), max(SIZE, round(height * factor))), Image.BICUBIC)
        notes.append("The image is smaller than 224 pixels and was enlarged, which makes the result less reliable.")

    positions = tile_positions(*img.size)
    tiles = np.stack([np.asarray(img.crop((x, y, x + SIZE, y + SIZE))) for x, y in positions])
    m1, m2 = models.score_tiles(tiles)
    raw = {"M1": m1, "M2": m2, "mean": (m1 + m2) / 2}
    thresholds = models.config.get("thresholds", {"M1": 0.5, "M2": 0.5, "mean": 0.5})
    main_name = models.config["main"]
    main_tiles = raw[main_name]
    score = float(calibrate(main_tiles.mean(), thresholds[main_name]))
    centre = tiles[len(tiles) // 2]

    return {
        "probability_ai": score,
        "verdict": "AI-generated" if score >= 0.5 else "Real photograph",
        "detectors": {
            models.config["names"][m]: float(calibrate(raw[m].mean(), thresholds[m])) for m in ["M1", "M2"]
        },
        "tiles": [{"x": x, "y": y, "size": SIZE, "probability_ai": float(calibrate(p, thresholds[main_name]))}
                  for (x, y), p in zip(positions, main_tiles)],
        "analysed_size": list(img.size),
        "original_size": [width, height],
        "closest_sources": models.attribute(centre),
        "notes": notes,
        "seconds": round(time.time() - start, 2),
    }


@app.get("/api/info")
def info():
    return {"model": models.config.get("description", ""), "max_tiles": MAX_TILES}


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")
