# CMDR-IAD: Cross-Modal Mapping and Dual-Branch Reconstruction for 2D–3D Industrial Anomaly Detection 

This repository provides the official PyTorch implementation of **CMDR-IAD**, using a real world polyurethane dataset. 

---

## 📑 Table of Contents
- [Dataset and Preprocessing](#Dataset)
- [Dataset Download](#Download)
- [Checkpoints](#checkpoints)
- [Code](#code)
- [Contacts](#contacts)

## Dataset
The polyurethane cutting dataset was collected within the MOROSAI project using a dual-sensor profilometer equipped with a 405\,nm laser. The system captures dense 3D point clouds focused on the cutting edges of large polyurethane components. As only geometric data is available, this dataset provides a realistic benchmark for evaluating the 3D-only inference mode of CMDR--IAD.

## Download

We evaluate CMDR-IAD on the **[MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)** dataset, which provides paired RGB images and 3D point clouds for industrial anomaly detection.

The raw dataset requires preprocessing to obtain aligned RGB images and organized point clouds. The necessary preprocessing scripts are provided in the `processing` directory.



## 📦 checkpoints

We release the pretrained CMDR-IAD checkpoints used to obtain the results reported in the paper.
The weight is provided to can be directly used for inference.

- The pretrained CMDR-IAD checkpoints used to obtain the results reported in the paper will be released after the paper is accepted.
- Create a folder named `checkpoints` in the project directory;
- Copy the downloaded weights into the `checkpoints`.

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
