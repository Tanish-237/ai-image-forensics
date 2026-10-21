# Detecting and Attributing AI-Generated Images

**B.Tech VII Semester Mini Project, CSE Department, MNNIT Allahabad (2026-27)**
**Mentor:** Dr. Anuja Dixit
**Team (Group 9):** Tanish Mittal (20233288), Vanshika Gautam (20233305), Shashank Kanaujiya (20233259), Utkarsh Raj Singh (20233298), Ashish Meena (20223058)

This project is a forensic study of AI-generated images. Given only the pixels, we try to tell whether a camera took an image or a generative model made it, and if a model made it, which one. We treat the generator as a *source* that leaves a statistical fingerprint, the same way a camera sensor does.

## Research questions

| # | Question | Experiment |
|---|---|---|
| RQ1 | Can a detector separate real photos from AI images made by known generators? | E1 |
| RQ2 | Does it still work on generators it never saw during training? | E2 |
| RQ3 | How robust is it to JPEG compression, resizing and blur? | E3 |
| RQ4 | Can it identify *which* generator made an image? | E4 |

## Methods

- **M1:** a ResNet-50 fine-tuned for real vs fake, following Wang et al., CVPR 2020.
- **M2:** frozen CLIP image features with a linear classifier, following Ojha et al., CVPR 2023.
- **M3:** NPR (neighbouring pixel relationships) with a shallow ResNet, following Tan et al., CVPR 2024.
- **M4:** the average of the M2 and M3 scores (no extra training).

## Results so far

| ID | What | Result |
|---|---|---|
| E0 | Pilot set (5 iPhone photos, 5 ChatGPT images): can file properties alone separate the classes? | Yes. Format, size, EXIF and colour-profile rules each score **100%**. |
| E0b | Same rules after normalization (crop to 512, rebuild from pixels, PNG) | Every rule drops to **50%** (chance). |
| E1 | Train and test on all six generators | M1 96.5%, M2 97.5%, M3 96.6%, **M4 98.8%** accuracy (ROC-AUC 0.994 / 0.997 / 0.994 / 0.999) |
| E2 | Train on SD15 only, test on five unseen generators | Mean detection on unseen generators: M1 54.6%, **M2 80.1%**, M3 59.6% |
| E5 | Train M1 on uncleaned data (fakes not JPEG-compressed), test on both versions | 98.8% on its own test set, **78.9%** once fakes are compressed like real photos (cleaned-data model: 96.5%) |
| E3 | JPEG, resizing and blur applied to the test set (no retraining) | Real photos get flagged as fake: at half size only 1–6% of real photos are called real. NPR is the most JPEG-sensitive. |
| E4 | Which of 7 sources made the image? (chance 14.3%) | CLIP + LR 80.0%, **fine-tuned ResNet-50 86.1%**; most confusion between SD15 and Wukong |
| E6 | Same detectors on 2024–26 generators (OpenFake: GPT Image 1.5/2, Midjourney 7, Flux.2, Nano Banana Pro, …) | ROC-AUC falls from 0.99+ to **0.72–0.75**; GPT Image 2 detected at only 14–34% |
| E6b | 100 OpenFake images (50 real, 50 AI) sent through WhatsApp | ROC-AUC falls from 0.67–0.70 to **0.46–0.58** (near chance); more false alarms *and* more missed fakes |
| E7 | Training on 2026 data | In progress |

## Repository layout

```
notebooks/   Colab notebooks, one per step (run in order)
src/         Reusable Python code imported by the notebooks
report/      Final report (MNNIT LaTeX template): thesis.tex, references.bib
docs/        Literature notes and dataset documentation
results/     Small result files (CSV/JSON) and figures; no raw data
```

Raw data is **not** stored in this repo. Pilot images live on Google Drive (`MyDrive/minor-project/`). The main dataset, [Tiny GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage), downloads automatically in the notebooks.

## Running the notebooks

1. Open a notebook in Google Colab: **File > Open notebook > GitHub**, then paste this repo's URL.
2. Go to **Runtime > Change runtime type > T4 GPU**.
3. Run the cells from top to bottom.

## Building the report

```bash
cd report
tectonic thesis.tex     # produces thesis.pdf
```

## Dataset and licences

Tiny GenImage is a subset of GenImage (Zhu et al., NeurIPS 2023) and is distributed under CC-BY-NC-SA-4.0. We use it for academic, non-commercial work only.
