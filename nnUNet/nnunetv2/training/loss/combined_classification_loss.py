import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedSegmentationClassificationLoss(nn.Module):
    def __init__(self, seg_loss, classification_weight=1.0, epoch=0):
        super().__init__()
        self.seg_loss = seg_loss
        self.class_loss = nn.CrossEntropyLoss()
        self.classification_weight = classification_weight
        self.epoch = epoch
    def forward(self, output, target):
        seg_output = output['seg']
        class_output = output['class']
        seg_target = target['target']
        class_target = target['class']
        loss_seg = self.seg_loss(seg_output, seg_target)
        loss_cls = self.class_loss(class_output, class_target)
        if self.epoch < 10:
            self.classification_weight =  0.05 
        elif self.epoch < 30:
            self.classification_weight = 0.15
        else:
            self.classification_weight = 0.3
            

        return (1-self.classification_weight)*loss_seg + self.classification_weight * loss_cls
