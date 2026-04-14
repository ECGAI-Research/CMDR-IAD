# CMDR-IAD: Cross-Modal Mapping and Dual-Branch Reconstruction for 2D–3D Industrial Anomaly Detection

This repository provides the official PyTorch implementation of **CMDR-IAD**, an unsupervised framework for industrial anomaly detection that integrates RGB appearance and 3D surface geometry.

---

## 📑 Table of Contents
- [Introduction](#introduction)
- [Cite this work](#cite-this-work)
- [Datasets](#datasets)
- [Checkpoints](#checkpoints)
- [Code](#code)
- [Contacts](#contacts)

## Introduction
Multimodal industrial anomaly detection remains challenging under noisy or incomplete modalities, where existing approaches based on memory banks, teacher--student frameworks, or naive fusion struggle to effectively exploit complementary information across 2D and 3D data. To address these limitations, this paper presents CMDR--IAD, a unified and novel framework for unsupervised multimodal anomaly detection that advances multi-view representation learning by jointly modeling heterogeneous 2D--3D data through bidirectional cross-modal alignment and dual-branch reconstruction. We propose two key components: (i) a reliability-gated mapping anomaly that captures cross-view discrepancies, and (ii) a confidence-weighted reconstruction anomaly that models intra-modal deviations. These signals are computed from independently optimized cross-modal mapping and reconstruction modules, and are adaptively fused to produce accurate anomaly localization under incomplete, noisy, or partially aligned multimodal inputs. CMDR--IAD achieves 97.3% image-level AUROC, 99.6% pixel-level AUROC, and 97.6% AUPRO on the MVTec 3D-AD benchmark. On a real-world polyurethane dataset, the 3D-only variant attains 92.6% I-AUROC and 92.5% P-AUROC, demonstrating robustness and flexibility across multimodal and unimodal settings. These results highlight the effectiveness of CMDR--IAD for multimodal representation learning and industrial anomaly detection.
<image src="Architectures/CMDR-IAD.png">

**Figure:** Overview of the CMDR-IAD architecture. The framework learns cross-modal mappings between RGB and 3D features and uses dual-branch reconstruction with adaptive fusion for anomaly detection.

## Cite this work
🖋️ If you find this code useful in your research, please cite:

```bibtex
@misc{daci2026crossmodalmappingdualbranchreconstruction,
      title={Cross-Modal Mapping and Dual-Branch Reconstruction for 2D-3D Multimodal Industrial Anomaly Detection}, 
      author={Radia Daci and Vito Renò and Cosimo Patruno and Angelo Cardellicchio and Abdelmalik Taleb-Ahmed and Marco Leo and Cosimo Distante},
      year={2026},
      eprint={2603.03939},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.03939}, 
}
```
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
CMDR-IAD provides scripts for **training** and **inference** of cross-modal mapping and dual-branch reconstruction networks for industrial anomaly detection.

To train CMDR-IAD, use the train.py script.
To Test CMDR-IAD. use the inference.py script.


Train and test options

`--dataset_path` : Path to the root directory of the MVTec 3D-AD dataset.

`--checkpoint_savepath` : Directory where trained checkpoints are saved (training) or read from (inference) (default: `./checkpoints/CMDR_IAD_checkpoints`).

`--class_name` : Object category to train on or to test on .

`--epochs_no` : Number of epochs.

`--batch_size` : Batch size.

Each object category is trained independently, and the resulting checkpoints are stored per class for inference.

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
