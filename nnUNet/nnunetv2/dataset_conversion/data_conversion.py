import os
import shutil
import csv
from glob import glob
from tqdm import tqdm
import json



from nnunetv2.paths import nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import save_json, join

# === CONFIGURATION ===
SOURCE_DIR = "/content/drive/MyDrive/ML-Quiz-3DMedImg"  # Root of the original dataset
TARGET_DATASET_ID = "Dataset050_MyProject"
DATASET_NAME = "PancreasMultiTask"

target_dir = os.path.join(nnUNet_raw, TARGET_DATASET_ID)
imagesTr = os.path.join(target_dir, "imagesTr")
labelsTr = os.path.join(target_dir, "labelsTr")
imagesTs = os.path.join(target_dir, "imagesTs")
maybe_mkdir_p(imagesTr)
maybe_mkdir_p(labelsTr)
maybe_mkdir_p(imagesTs)

classification_rows = []
training_list = []
val_list = []

# === PROCESS TRAIN + VAL ===
def process_split(split_folder, mode="train"):
    subtype_folders = sorted(glob(os.path.join(SOURCE_DIR, split_folder, "subtype*")))
    for subtype_path in subtype_folders:
        subtype_label = int(os.path.basename(subtype_path).replace("subtype", ""))
        nii_files = sorted(glob(os.path.join(subtype_path, "*_0000.nii.gz")))
        for image_path in nii_files:
            base = os.path.basename(image_path).replace("_0000.nii.gz", "")
            mask_path = os.path.join(subtype_path, base + ".nii.gz")
            case_id = base

            image_target = os.path.join(imagesTr, f"{case_id}_0000.nii.gz")
            mask_target = os.path.join(labelsTr, f"{case_id}.nii.gz")
            shutil.copy(image_path, image_target)
            shutil.copy(mask_path, mask_target)

            classification_rows.append([case_id, subtype_label])

            if mode == "train":
                training_list.append(case_id)
            elif mode == "validation":
                val_list.append(case_id)

process_split("train", mode="train")
process_split("validation", mode="validation")

# === PROCESS TEST SET ===
test_files = sorted(glob(os.path.join(SOURCE_DIR, "test", "*_0000.nii.gz")))
for test_path in test_files:
    base = os.path.basename(test_path).replace("_0000.nii.gz", "")
    case_id = f"test_{base}"
    image_target = os.path.join(imagesTs, f"{case_id}_0000.nii.gz")
    shutil.copy(test_path, image_target)

# === SAVE CLASSIFICATION LABELS ===
csv_path = os.path.join(target_dir, "classification_labels.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Case", "Label"])
    writer.writerows(classification_rows)

# === GENERATE dataset.json ===
generate_dataset_json(
    output_folder=target_dir,
    channel_names={0: "CT"},
    labels={
        "background": 0,
        "pancreas": 1,
        "lesion": 2
    },
    num_training_cases=len(training_list),
    file_ending=".nii.gz",
    dataset_name=DATASET_NAME,
    description="Multi-task pancreas segmentation and subtype classification using CT images.",
    converted_by="Your Name Here",
    license="CC-BY 4.0"
)

# === CREATE splits_final.json for fixed validation split ===
splits_final = [{"train": training_list, "val": val_list}]
save_json(splits_final, os.path.join(target_dir, 'splits_final.json'), sort_keys=False)

print(f"✅ Dataset conversion completed. Saved to {target_dir}")
