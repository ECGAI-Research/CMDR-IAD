# CMDR-IAD

**CMDR-IAD: Cross-Modal Mapping and Dual-Branch Reconstruction for 2D–3D Industrial Anomaly Detection**

This repository provides the official PyTorch implementation of **CMDR-IAD**, an unsupervised framework for industrial anomaly detection that integrates RGB appearance and 3D surface geometry.  
The method is lightweight, does not rely on memory banks or teacher–student architectures, and supports multimodal (2D+3D) as well as single-modality (2D-only or 3D-only) settings.

---

## 📌 Overview

Multimodal industrial anomaly detection benefits from combining complementary RGB and 3D information. However, existing unsupervised approaches often depend on memory banks, teacher–student schemes, or fragile fusion strategies, which can degrade performance under noisy depth, weak texture, or missing modalities.
<image src="Architectures/CMDR-IAD_Architecture.jpg">

---

## 📑 Table of Contents
- [Introduction](#introduction)
- [Datasets](#datasets)
- [Checkpoints](#checkpoints)
- [Code](#code)
- [Contacts](#contacts)

## Introduction
CMDR-IAD is an unsupervised, modality-flexible framework for industrial anomaly detection. It models cross-modal relationships between RGB and 3D features and learns dual-branch reconstructions of normal appearance and geometry.

Key Features:

- Cross-modal mapping: projects 2D features into 3D space and vice versa.

- Dual-branch reconstruction: learns modality-specific normal patterns for RGB and 3D data.

- Adaptive fusion: combines reconstruction errors and cross-modal discrepancies with spatial reliability weighting, producing accurate and stable anomaly maps.

- Lightweight design using frozen encoders and small mapping/decoding networks for fast inference and low memory usage.

- 3D-only mode: the 3D reconstruction branch works independently if only point clouds are available.

<p align="center">
  <img src="images/architecture.png" width="800">
</p>

**Figure:** Overview of the CMDR-IAD architecture. The framework learns cross-modal mappings between RGB and 3D features and uses dual-branch reconstruction with adaptive fusion for anomaly detection.


## Datasets

We evaluate CMDR-IAD on the **[MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)** dataset, which provides paired RGB images and 3D point clouds for industrial anomaly detection.

The raw dataset requires preprocessing to obtain aligned RGB images and organized point clouds. The necessary preprocessing scripts are provided in the `processing` directory.



## 📦 checkpoints

We release the pretrained CMDR-IAD checkpoints used to obtain the results reported in the paper.
The weights are provided per object category and can be directly used for inference.

- The pretrained CMDR-IAD checkpoints used to obtain the results reported in the paper will be released after the paper is accepted.
- Create a folder named `checkpoints` in the project directory;
- Copy the downloaded weights into the `checkpoints`.

## Code
To train CMDR-IAD, use the train.py script.
The training procedure independently optimizes the cross-modal mapping networks and the dual-branch reconstruction modules for a given object category, following the protocol described in the paper.
Training command
python train.py --class_name tire

Training options

`--dataset_path` : Path to the root directory of the MVTec 3D-AD dataset.

`--checkpoint_savepath` : Directory where trained checkpoints will be saved (default: `./checkpoints/CMDR_IAD_checkpoints`).

`--class_name` : Object category to train on.

`--epochs_no` : Number of training epochs.

`--batch_size` : Batch size.

Each object category is trained independently, and the resulting checkpoints are stored per class.

## Contacts
For questions, please send an email to <radia.daci@isasi.cnr.it>. .

## 📂 Repository Structure

```text
CMDR-IAD/
├── networks/
│   ├── features.py
│   ├── Map.py
│   ├── Dec2d.py
│   ├── Dec3d.py
│   ├── dataset.py
│   └── full_models.py
├── processing/
│   ├── aggregate_results.py
│   ├── preprocess_mvtec.py
├── utils/
│   ├── mvtec3d_utils.py
│   ├── pointnet2_utils.py
│   ├── metrics_utils.py
│   └── general_utils.py
│
├── train.py
├── inference.py
├── README.md
├── requirements.txt
