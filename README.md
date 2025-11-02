# CNN-Based Image Classification Using Transfer Learning (Materials Science / LPBF SDSS 2507)

This repository shows how to use **convolutional neural networks (CNNs)** with **transfer learning** (GoogLeNet) in **MATLAB** to classify microstructural or microscopy images from **additively manufactured (LPBF) AISI 2507 super duplex stainless steel**.

The goal is to provide a **reproducible, materials-oriented deep learning example**: starting from raw microstructure images and ending with a trained model that can tell apart different processing or heat-treatment conditions.

---

## 1. Background / Motivation

Additively manufactured duplex / super duplex stainless steels exhibit microstructures that vary with:
- laser processing window (power, speed, hatch spacing),
- build orientation,
- and especially **post-processing / heat treatment**.

These changes appear as **visually separable patterns** under SEM/optical imaging. A CNN with transfer learning is a good way to:
1. turn these images into labeled data,
2. learn robust features from a pretrained network (GoogLeNet),
3. and test how well the network can discriminate between conditions such as `AS`, stress-relieved, or solution-annealed samples.

This repo demonstrates exactly that.

---

## 2. What’s Inside

- **`README.md`** — this file.
- **`Transfer Learning(GoogLeNet) for Microstructure Classification of Additively Manufactured AISI 2507 SDSS Samples`** — MATLAB script/notebook illustrating the full transfer-learning workflow (load images → split → augment → retrain final layers → evaluate).  
- **`LICENSE`** — MIT, so you can reuse and extend.  
- (Optional / to be added) **`/dataset/`** or similar folder for image data grouped by class.

> Note: GitHub currently shows the project as “Digital Image Processing & Deep Learning for Material Science,” so you can extend this repo later with other course projects. :contentReference[oaicite:2]{index=2}

---

## 3. Requirements

- **MATLAB** (R2022a or newer recommended)
- **Deep Learning Toolbox**
- **Image Processing Toolbox** (for reading/augmenting microscopy images)
- **Computer Vision Toolbox** (optional but useful)
- A **GPU** will speed up training, but CPU works for small datasets.

---

## 4. Dataset Expectations

This project assumes your images are organized in **subfolders by class**, e.g.

```text
data/
├── AS/
│   ├── img_001.png
│   ├── img_002.png
├── SR400_1h/
│   ├── img_010.png
├── SR500_1h/
│   ├── ...
└── SA1100_15min/
    ├── ...

imds = imageDatastore('data', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

