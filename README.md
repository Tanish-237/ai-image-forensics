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
| RQ3 | How robust is it to JPEG compression, resizing and cropping? | E3 |
| RQ4 | Can it identify *which* generator made an image? | E4 |

## Methods

- **M1:** a ResNet-50 fine-tuned for real vs fake, following Wang et al., CVPR 2020.
- **M2:** frozen CLIP image features with a linear classifier, following Ojha et al., CVPR 2023.

## Results so far

| ID | What | Result |
|---|---|---|
| E0 | Pilot set (5 iPhone photos, 5 ChatGPT images): can file properties alone separate the classes? | Yes. Format, size, EXIF and colour-profile rules each score **100%**. |
| E0b | Same rules after normalization (crop to 512, rebuild from pixels, PNG) | Every rule drops to **50%** (chance). |
| E1–E4 | Main experiments on Tiny GenImage | In progress |

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
