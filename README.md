# Multi-task Learning for Pancreas Segmentation and Subtype Classification using nnUNetv2

## Project Description

This project extends nnUNetv2 for pancreas segmentation and lesion subtype classification in 3D CT scans by adding a classification head to the standard model.

## Installation

To install the project in editable mode, run:

```bash
pip install -e .
```

## Training

After preprocessing your dataset into the nnUNetv2 format, you can start training with the following command:

```bash
nnUNetv2_train Dataset050_MyProject 3d_fullres 0 -tr nnUNetTrainerWithClassification -p nnUNetResEncUNetMPlans
```

- `Dataset050_MyProject`: Replace with your dataset ID if different.
- `nnUNetTrainerWithClassification`: Custom trainer for joint segmentation and classification.
- `nnUNetResEncUNetMPlans`: Model configuration based on ResEncUNet plans.
