"""Publishes the web page (without the Python backend) as a free static Hugging Face Space."""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO = "tanish-237/ai-image-forensics"
HERE = Path(__file__).parent

README = """---
title: AI Image Forensics
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: static
pinned: false
license: cc-by-nc-4.0
short_description: Detecting and attributing AI-generated images
---

Web page of our B.Tech minor project at MNNIT Allahabad. The findings are always available; live analysis works while our detector server is running.
"""

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    shutil.copytree(HERE / "static", out / "static", ignore=shutil.ignore_patterns(".DS_Store"))
    shutil.copy(HERE / "static" / "index.html", out / "index.html")
    (out / "README.md").write_text(README)
    api = HfApi()
    api.create_repo(REPO, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=REPO, repo_type="space", commit_message="Update page")
    print(f"Published: https://huggingface.co/spaces/{REPO}")
