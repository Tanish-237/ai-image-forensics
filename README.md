# Detecting and Attributing AI-Generated Images

A forensic study of AI-generated images: given an image, decide whether a camera took it or a generative model made it, and if a model made it, which one. We treat the generator as a *source* that leaves a statistical fingerprint, in the same way a camera sensor does, and test how well current detectors work on new generators, on processed images and on images shared through WhatsApp.

B.Tech VII Semester Minor Project, Department of Computer Science and Engineering, MNNIT Allahabad (2026-27).
Mentor: Dr. Anuja Dixit.
Team (Group 9): Tanish Mittal (20233288), Vanshika Gautam (20233305), Shashank Kanaujiya (20233259), Utkarsh Raj Singh (20233298), Ashish Meena (20223058).

- **Report:** [`report/thesis.pdf`](report/thesis.pdf)
- **Website:** https://tanish-237-ai-image-forensics.static.hf.space (live analysis while our detector server is running)
- **Trained models:** https://huggingface.co/tanish-237/ai-image-forensics-models

## Research questions

| # | Question | Experiments |
|---|---|---|
| RQ1 | Can a detector separate real photographs from AI images made by known generators? | E1, E5 |
| RQ2 | Does it still work on generators it never saw during training? | E2, E6, E7, E8 |
| RQ3 | How do JPEG compression, resizing, blur and WhatsApp sharing affect it? | E3, E6, E8, E10 |
| RQ4 | Can it identify *which* generator made an image? | E4, E11, E12 |

## Methods

| | Detector | Based on |
|---|---|---|
| M1 | ResNet-50 fine-tuned for real vs generated | Wang et al., CVPR 2020 |
| M2 | Frozen CLIP ViT-L/14 features with logistic regression | Ojha et al., CVPR 2023 |
| M3 | NPR (neighbouring pixel relationships) with a shallow ResNet | Tan et al., CVPR 2024 |
| M4 | Average of the M2 and M3 scores (no extra training) | |

The final detector (E10) is M2 trained on current generators and real photographs from 57 phone models. It scores a grid of 224 x 224 tiles at the image's original resolution and averages the tile scores.

## Results

| ID | Experiment | Main result |
|---|---|---|
| E0 | Pilot set: 5 iPhone photos, 5 ChatGPT images | File format, size and EXIF rules each separate the classes perfectly; after normalisation every rule falls to 50% |
| E1 | Train and test on six Tiny GenImage generators (de-biased) | Accuracy M1 96.5%, M2 97.5%, M3 96.6%, M4 98.8% |
| E2 | Train on SD15 only, test on five unseen generators | Mean detection M1 54.6%, **M2 80.1%**, M3 59.6% |
| E3 | JPEG, resizing and blur on the test set | Processing makes real photos look generated: at half size only 1-6% are called real |
| E4 | Which of seven sources made the image? (chance 14.3%) | CLIP 80.0%, **fine-tuned ResNet-50 86.1%** |
| E5 | Same detector trained on uncorrected data | 98.8% on its own test set, **78.9%** once the AI images are compressed like the real ones |
| E6 | 2021-23 detectors on 2024-26 generators (OpenFake), and 100 images sent through WhatsApp | ROC-AUC falls to **0.72-0.75**; after WhatsApp close to chance |
| E7 | Retrain on OpenFake's training split (72 generators, 2022-25) | ROC-AUC on unseen current generators: M2 0.720 to **0.945** |
| E8 | More data, augmentation, full-image tile analysis | M2 reaches 0.957; tiles raise ROC-AUC after WhatsApp from 0.846 to **0.933**; augmentation did not help |
| E9 | Deployment detector including the newest generators | ROC-AUC 0.999 on OpenFake, but only **45%** of Unsplash phone photos and 28% of camera photos called real |
| E10 | E9 plus phone photos from 57 models, ChatGPT and Gemini images | **98%** of phone photos called real; 97% ChatGPT and 99% Gemini images detected |
| E11 | Attribution to nine current generator families (chance 11%) | 67% on benchmark images; did not transfer to our own app images (5 of 62) |
| E12 | Map of the CLIP feature space (t-SNE, 10-nearest-neighbour agreement) | Real and AI images separate clearly (96% / 94% agreement, chance 35% / 65%) |
| Own test | 62 files we made with ChatGPT, Gemini and Grok (originals, screenshots, WhatsApp copies) | **57 of 62** called AI-generated; every original identified, most by its Content Credentials |

The website also checks signed **Content Credentials (C2PA)**, which ChatGPT and Gemini attach to their images. A valid record decides the verdict; otherwise the pixel detector does, and scores between 40% and 60% are reported as uncertain. All numbers above are our own measurements; details are in the report.

## Repository structure

```
notebooks/         Colab notebooks, one per experiment, numbered in the order they were run
website/           FastAPI backend (app.py, provenance.py) and the web page (static/)
  trust/           Official C2PA trust list used to verify Content Credentials
report/            Final report: thesis.tex, references.bib, figures/, thesis.pdf
results/           Small result files produced on the laptop (own test, tile tests)
data_collection/   Prompts used to generate our own test images
```

Datasets and intermediate results (features, checkpoints, CSV summaries) are stored on Google Drive under `MyDrive/minor-project/`, not in this repository.

## Notebooks

| Notebook | Content |
|---|---|
| 01_pilot_inspection | Pilot set inspection and normalisation (E0) |
| 02_dataset_inventory | Tiny GenImage inventory |
| 03_preprocess | Bias removal and train/validation/test splits |
| 04_train_detector | M1: ResNet-50 (E1) |
| 05_fingerprints | Noise-residual spectra and NPR residuals |
| 06_clip_detector, 07_npr_detector | M2 and M3 (E1) |
| 08a, 08b, 08c | Unseen generators: each detector trained on SD15 only (E2) |
| 09_compare_detectors | Comparison and M4 |
| 10_e5_shortcut | Effect of de-biasing (E5) |
| 11_e3_robustness | JPEG, resizing and blur (E3) |
| 12_e4_attribution | Seven-way attribution (E4) |
| 13_e6_openfake, 14_e6_whatsapp | Current generators and WhatsApp sharing (E6) |
| 15_e7_train_openfake | Training on current generators (E7) |
| 16_demo | Early interactive demo |
| 17_e8_robust_training | More data, augmentation, full-image analysis (E8) |
| 18_export_website_models | Export of the website models to Hugging Face |
| 19_deploy_detector | Deployment detector (E9) |
| 20_real_world_data | Real-world photographs and current generators (E10) |
| 21_attribution_current | Attribution of current generator families (E11) |
| 22_fingerprint_map | Feature-space map (E12) |

To run a notebook, open it in Google Colab (**File > Open notebook > GitHub**), select a T4 GPU runtime and run the cells in order. Each notebook mounts Google Drive and reads the outputs of the notebooks before it.

## Website

```bash
website/run.sh            # local site at http://localhost:7860
website/run.sh --public   # also opens a temporary public link (needs cloudflared)
```

The first run creates a Python environment and downloads the models; later runs also work offline. An upload is checked in two steps: its provenance (C2PA Content Credentials verified against the official trust list, and AI labels in the metadata) and its pixels. See [`website/README.md`](website/README.md) for details. The hosted page is updated with `python website/publish_static.py`.

## Building the report

```bash
cd report
tectonic thesis.tex
```

## Data and licences

| Data | Licence | Use |
|---|---|---|
| [Tiny GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) (subset of GenImage, Zhu et al., NeurIPS 2023) | CC BY-NC-SA 4.0 | E1-E5 |
| [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (Livernoche et al., 2025) | CC BY-NC 4.0 | E6-E12 |
| Wikimedia Commons phone photographs | individual CC licences | E10 (features only) |
| Unsplash Lite | Unsplash Lite licence | E10 (features only) |
| ShareGPT-4o-Image | Apache 2.0 | E10 (features only) |
| bitmind/nano-banana | MIT | E10 (features only) |
| C2PA trust list | CC BY 4.0 | website |

No dataset images are redistributed here; the sample images on the website were generated or photographed by us. The project is for academic, non-commercial use.
