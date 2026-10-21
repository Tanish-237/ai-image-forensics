---
title: AI Image Forensics
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-4.0
short_description: Is this image AI-generated? Detection and fingerprints.
---

# AI Image Forensics

Website of our B.Tech minor project at MNNIT Allahabad, *Detecting and Attributing AI-Generated Images: A Forensic Fingerprinting Study*.

Upload an image to see whether our detectors consider it AI-generated. The image is analysed at full resolution in a grid of 224 x 224 tiles by two detectors trained on recent generators: a fine-tuned ResNet-50 and frozen CLIP features with a linear classifier. The page also summarises the results of our experiments.

- `app.py`: FastAPI backend (`POST /api/analyze`)
- `static/`: the web page
- Model weights are downloaded at start-up from `tanish-237/ai-image-forensics-models`.

Run locally:

```
pip install torch torchvision -r requirements.txt
MODEL_DIR=/path/to/models uvicorn app:app --port 7860
```
