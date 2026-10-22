# Website

Web page and backend of the project. A user uploads an image (JPEG, PNG, WebP or HEIC, up to 25 MB) and gets a verdict together with the evidence behind it. Uploaded images are analysed in memory and not stored.

## How an upload is analysed

1. **Provenance** (`provenance.py`). The file's C2PA Content Credentials are read with `c2pa-python` and verified against the official C2PA trust list in `trust/`, without network access. A valid record stating that a trained model created the image decides "AI-generated"; a record of a camera capture is accepted only from a signer on the trust list. Unsigned AI labels (IPTC digital source type, Stable Diffusion or ComfyUI settings, generator names in the EXIF software tag) are also used. Camera EXIF tags are shown but never decide.
2. **Pixels** (`app.py`). Without usable provenance, the image is covered by a grid of 224 x 224 tiles at its original resolution (16 by default; 9 or 25 selectable). Each tile is embedded with frozen CLIP ViT-L/14 and scored by the E10 logistic-regression detector; the tile scores are averaged and rescaled so that the calibrated threshold maps to 50%. Scores between 40% and 60% are reported as uncertain.

The page also offers a simulated WhatsApp share (longest side at most 1600 pixels, JPEG quality 80, metadata removed) with a side-by-side comparison, six sample images from our own test, and a QR code of the public link while one is open.

## Files

| File | Purpose |
|---|---|
| `app.py` | FastAPI backend: `POST /api/analyze`, `GET /api/info`, `GET /api/qr` |
| `provenance.py` | Content Credentials, metadata labels and camera EXIF |
| `static/` | The web page (`index.html`, `app.js`, `style.css`), figures and sample images |
| `trust/` | C2PA trust list (CC BY 4.0) and the certificate usage settings |
| `run.sh` | Starts the backend on a laptop, optionally with a public cloudflared link |
| `publish_static.py` | Publishes the page without the backend as a static Hugging Face Space |
| `Dockerfile`, `requirements.txt` | Container build for hosting the backend |

The model weights and `config.json` are downloaded at start-up from [tanish-237/ai-image-forensics-models](https://huggingface.co/tanish-237/ai-image-forensics-models); `MODEL_DIR` can point to a local copy instead.

## Running

```bash
./run.sh            # http://localhost:7860
./run.sh --public   # also prints a temporary public link
```

Environment variables: `PORT` (default 7860), `MAX_TILES` (default tile count, 16), `MODEL_DIR` (local model folder), `SHOW_FAMILIES=1` (show the E11 family guess, off by default because it did not transfer to our own images).

The hosted static page analyses images only when opened with `?api=<public link>`, which `run.sh --public` prints.
