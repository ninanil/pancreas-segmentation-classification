import os
import pandas as pd
import torch
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2 

class nnUNetDatasetWithClassification(nnUNetDatasetBlosc2):
    def __init__(self, folder, identifiers=None, folder_with_segs_from_previous_stage=None):
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)

        # Load classification labels
        # raw_folder = os.path.dirname(folder).replace('nnUNet_preprocessed', 'nnUNet_raw')

        # csv_path = os.path.join(raw_folder, 'classification_labels.csv')
        # Instead of going back to nnUNet_raw, read from same folder
        folder_path = os.path.dirname(folder)
        csv_path = os.path.join(folder_path, 'classification_labels.csv')  # UPDATED

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Expected classification CSV at: {csv_path}")

        df = pd.read_csv(csv_path)
        df["Case"] = df["Case"].str.strip()  # Strip whitespace from case names
        self.class_label_map = dict(zip(df["Case"], df["Label"]))
        self.identifiers = [id_ for id_ in self.identifiers if id_ in self.class_label_map]
        


    def __getitem__(self, i):
        
        
        case_id = self.identifiers[i]
        
    
        if case_id not in self.class_label_map:
            raise KeyError(f"❌ Missing label for: {case_id}")
    
        class_label = self.class_label_map[case_id]
        data, seg, seg_prev, properties = self.load_case(case_id)
    
        return {
            'data': torch.tensor(data).float(),
            'seg': torch.tensor(seg).long(),
            'seg_prev': torch.tensor(seg_prev).long() if seg_prev is not None else None,
            'properties': properties,
            'class': torch.tensor(class_label).long()
        }

        # # used during inference/debugging
        # data, seg, seg_prev, properties = self.load_case(self.identifiers[i])
        # case_id = self.identifiers[i]
        # class_label = self.class_label_map[case_id]

        # return {
        #     'data': torch.tensor(data).float(),
        #     'seg': torch.tensor(seg).long(),
        #     'seg_prev': torch.tensor(seg_prev).long() if seg_prev is not None else None,
        #     'properties': properties,
        #     'class': torch.tensor(class_label).long()
        # }