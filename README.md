# Multi-task Learning for Pancreas Segmentation and Subtype Classification using nnUNetv2

## Project Description

This project extends nnUNetv2 for pancreas segmentation and lesion subtype classification in 3D CT scans by adding a classification head to the standard model.

## Environment and Requirements

- **OS**: Ubuntu 20.04 / Windows 10+
- **CPU**: Intel Core i7 or higher
- **RAM**: ≥ 16 GB
- **GPU**: NVIDIA T4 (Kaggle) or A100 (Recommended)
- **CUDA**: 11.8
- **Python**: 3.10+

## Installation

To install the project in editable mode, run:

```bash
pip install -e .
```

---

## 📂 Dataset

This project uses a private 3D CT dataset with manually labeled subtypes.

| Subtype    | Train | Val  |
|------------|-------|------|
| Subtype 0  | 62    | 9    |
| Subtype 1  | 106   | 15   |
| Subtype 2  | 84    | 12   |
| **Total**  | 252   | 36   |

### Folder Structure

Preprocessing requires the dataset to follow **nnUNetv2 format**:

```
Dataset050_MyProject/
├── imagesTr/
├── labelsTr/
└── dataset.json
```

---

## Preprocessing

Preprocessing includes:
- Intensity normalization
- Isotropic resampling
- Cropping (foreground region)

Run the following to preprocess the dataset:

```bash
nnUNetv2_plan_and_preprocess -d Dataset050_MyProject -p nnUNetPlannerResEncM
```

---

## Training

To train the joint segmentation and classification model:

```bash
nnUNetv2_train Dataset050_MyProject 3d_fullres 0 \
-tr nnUNetTrainerWithClassification \
-p nnUNetResEncUNetMPlans
```

- `Dataset050_MyProject`: Replace with your dataset ID
- `3d_fullres`: Full resolution 3D UNet training
- `nnUNetTrainerWithClassification`: Custom trainer with a classification head
- `nnUNetResEncUNetMPlans`: Residual Encoder UNet M configuration

The classification head can be configured to use features from:
- Last decoder output (C=32)
- First decoder output (C=320)
- Encoder output (C=320)
- Or multi-scale decoder features

---

## Inference

To generate predictions for test/val data:

```bash
nnUNetv2_predict -i <input_folder> -o <output_folder> \
-d Dataset050_MyProject -c 3d_fullres -f 0 \
-tr nnUNetTrainerWithClassification \
-p nnUNetResEncUNetMPlans
```

Example:

```bash
nnUNetv2_predict -i imagesTs/ -o preds/ -d Dataset050_MyProject -c 3d_fullres -f 0 \
-tr nnUNetTrainerWithClassification -p nnUNetResEncUNetMPlans
```

---
Metrics include:
- **DSC** (Dice Similarity Coefficient) for:
  - Whole pancreas (label > 0)
  - Lesion only (label == 2)
- **Classification**:
  - Accuracy
  - Macro F1 score

---
## Trained Models

> Coming soon — trained weights and checkpoints will be published here.

---

## 🧠 Acknowledgements

Built upon the excellent [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet) framework by the DKFZ team.
