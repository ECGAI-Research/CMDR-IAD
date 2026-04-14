# CMDR-IAD: Cross-Modal Mapping and Dual-Branch Reconstruction for 2D–3D Industrial Anomaly Detection 

This repository provides the official PyTorch implementation of **CMDR-IAD**, using a real world polyurethane dataset. 

---

## 📑 Table of Contents
- [Dataset and Preprocessing](#Dataset)
- [Dataset Download](#Download)
- [Checkpoints](#checkpoints)
- [Code](#code)
- [Contacts](#contacts)
## 📂 Dataset and Preprocessing

### Polyurethane 3D Dataset

The polyurethane cutting dataset was collected within the MOROSAI project using a dual-sensor profilometer equipped with a 405 nm laser. The system captures dense 3D point clouds focused on the cutting edges of large polyurethane components.

Since only geometric data is available (no RGB images), this dataset is used to evaluate the 3D-only inference mode of CMDR-IAD under realistic industrial conditions.

---

## ⚙️ Preprocessing Pipeline

The raw point clouds are processed using a three-stage pipeline:

### 1. Geometric Outlier Detection (Isolation Forest)

Given a point cloud:

P = {p_i in R^3}, i = 1,...,N

We apply an Isolation Forest to detect sparse geometric anomalies caused by:

* cutting defects
* sensor noise

Settings:

* contamination = 0.0001 (very low due to defect sparsity)

Output:

* binary point-level anomaly mask for each scan

---

### 2. Sequential Chunking and Labeling

Each point cloud is split into fixed-size chunks:

* 9216 points per chunk

For each chunk C_k, we compute the anomaly ratio:

f_k = (1 / |C_k|) * sum(m_i)

Where:

* m_i is the anomaly label from Isolation Forest

Chunk labeling rule:

* anomalous if f_k >= 0.0025
* otherwise normal

Stored data:

* chunked point clouds
* chunk-level labels
* point-wise anomaly masks

---

### 3. Dataset Construction and Normalization

All chunks are aggregated into a unified dataset.

Protocol:

* one-class training (only normal samples used for training)
* 90/10 split on normal data

Final dataset:

* Training: 1856 normal samples
* Test: 235 samples

  * 207 normal
  * 28 anomalous

Normalization:
Each chunk is independently scaled to the range [-1, 1].

---

## 📊 Usage in CMDR-IAD

This dataset is used to:

* train the 3D reconstruction branch
* evaluate purely geometric anomaly detection
* benchmark performance in real industrial conditions

## 📦 Checkpoints

We provide pretrained checkpoints for CMDR-IAD trained on the polyurethane 3D dataset.

These weights can be directly used for inference or as initialization for further training.

---

### 📥 Download

* The pretrained checkpoints will be released after the paper is accepted.
* Stay tuned for updates or contact the authors for early access.

---

### 📁 Setup

1. Create a folder named `checkpoints` in the project root:

```
mkdir checkpoints
```

2. Place the downloaded weights inside:

```
./checkpoints/polyurethane_cuts
```
## Code
CMDR-IAD provides scripts for **training** and **inference** of cross-modal mapping and dual-branch reconstruction networks for industrial anomaly detection.

To train CMDR-IAD, use the train.py script.
To Test CMDR-IAD. use the inference.py script.

Train and test options

`--dataset_path` : Folder containing train.npy, test.npy, etc.

`--checkpoint_savepath` : Directory where trained checkpoints are saved (training) or read from (inference) (default: `./checkpoints/polyurethane_cuts`).

`--epochs_no` : Number of epochs.

`--batch_size` : Batch size.


## Contacts
For questions, please send an email to <radia.daci@isasi.cnr.it>.
