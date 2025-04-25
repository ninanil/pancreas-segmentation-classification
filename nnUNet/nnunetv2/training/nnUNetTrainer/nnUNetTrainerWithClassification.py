import os
import warnings
from time import sleep, time
import multiprocessing
from typing import Tuple, Union, List
from inspect import signature
import numpy as np
import pandas as pd
import torch
import re
import shutil
from torch import nn, autocast
import torch.distributed as dist
from torch._dynamo import OptimizedModule
from sklearn.metrics import accuracy_score, f1_score
from medpy.metric.binary import dc
import wandb
from copy import deepcopy

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.nnunet_dataset_with_classification import nnUNetDatasetWithClassification
from nnunetv2.training.loss.combined_classification_loss import CombinedSegmentationClassificationLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictorWithClassification
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.configuration import default_num_processes
from nnunetv2.paths import nnUNet_preprocessed
from batchgenerators.utilities.file_and_folder_operations import save_json, load_json
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class ClassificationHead(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)  # raw logits

class UNetWithClassificationHead(nn.Module):
    def __init__(self, unet: nn.Module, num_classes: int):
        super().__init__()
        self.unet = unet
        self.num_classes = num_classes  
        self.classification_head = None  # initialized after forward

    def forward(self, x):
        if not hasattr(self.unet, 'encoder'):
            raise RuntimeError("Inner UNet must have encoder and decoder")

        features = self.unet.encoder(x)
        decoder = self.unet.decoder
        seg_outputs = []
        decoder_input = features[-1]

        classification_feature = None  # <-- will store first decoder output (C=320)

        for s in range(len(decoder.stages)):
            x = decoder.transpconvs[s](decoder_input)
            x = torch.cat((x, features[-(s + 2)]), dim=1)
            x = decoder.stages[s](x)

            if s == 0:
                classification_feature = x  # <-- first decoder output (C=320)

            if decoder.deep_supervision:
                seg_outputs.append(decoder.seg_layers[s](x))
            elif s == (len(decoder.stages) - 1):
                seg_outputs.append(decoder.seg_layers[-1](x))
        
            decoder_input = x
        
        seg_outputs = seg_outputs[::-1]
        x_seg = seg_outputs if decoder.deep_supervision else seg_outputs[0]

        # Initialize classification head if needed
        if self.classification_head is None:
            in_channels = classification_feature.shape[1]  # should be 320
            self.classification_head = ClassificationHead(
                input_channels=in_channels, 
                num_classes=self.num_classes, 
                hidden_dim=128  
            ).to(classification_feature.device)

        x_cls = self.classification_head(classification_feature)
        return x_seg, x_cls
        
    # Fix for compatibility with nnUNet
    @property
    def decoder(self):
        return self.unet.decoder

    @property
    def encoder(self):
        return self.unet.encoder

 

class nnUNetTrainerWithClassification(nnUNetTrainer):
    def __init__(self, *args, **kwargs):
        

        # Filter kwargs to match parent __init__ exactly
        parent_keys = signature(super().__init__).parameters
        clean_kwargs = {k: v for k, v in kwargs.items() if k in parent_keys}
        super().__init__(*args, **clean_kwargs)

        # Inject your dataset class once here
        self.dataset_class = nnUNetDatasetWithClassification
        self.initial_lr = 1e-4
        self.num_epochs = 500
        self.save_every = 50
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.patience = 15  # Stop if no improvement for 15 epochs
        self.min_epochs = 200
        

        
    def build_network_architecture(self, architecture_class_name: str,
                                arch_init_kwargs: dict,
                                arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                num_input_channels: int,
                                num_output_channels: int,
                                enable_deep_supervision: bool = True) -> nn.Module:
        # Build base nnU-Net segmentation model
        base_net = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=True
        )

        # Wrap it with my classification logic
        return UNetWithClassificationHead(
            base_net, num_classes=3  # adapt channels to your last feature map
        )

    def _build_loss(self):
      # Build standard segmentation loss
      loss = DC_and_CE_loss(
          {'batch_dice': self.configuration_manager.batch_dice,
          'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
          {}, weight_ce=1, weight_dice=1,
          ignore_label=self.label_manager.ignore_label,
          dice_class=MemoryEfficientSoftDiceLoss
      )

      if self.enable_deep_supervision:
          deep_supervision_scales = self._get_deep_supervision_scales()
          weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
          weights[-1] = 0
          weights = weights / weights.sum()
          loss = DeepSupervisionWrapper(loss, weights)

      return CombinedSegmentationClassificationLoss(loss, classification_weight=0.05, epoch = self.current_epoch)
    
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,) -> BasicTransform:
        return nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded, foreground_labels,
                                                       regions, ignore_label)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        # we need to disable mirroring here so that no mirroring will be applied in inference!
        rotation_for_DA, do_dummy_2d_data_aug, _, _ = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        initial_patch_size = self.configuration_manager.patch_size
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
      
    def get_tr_and_val_datasets(self):
        
        tr_keys, val_keys = self.do_split()  # dynamically get training/val split
    
        dataset_tr = nnUNetDatasetWithClassification(
            self.preprocessed_dataset_folder,
            tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        dataset_val = nnUNetDatasetWithClassification(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        return dataset_tr, dataset_val


    # def train_step(self, batch: dict) -> dict:
    
    #     data = batch['data']
    #     seg_target = batch['target']        # segmentation GT
    #     class_target = batch['class']       # classification GT
        
    #     # Check for empty class_target
    #     if class_target.numel() == 0:
    #         raise ValueError("ERROR: class_target is empty. Check your dataloader output!")
    
    #     data = data.to(self.device, non_blocking=True)
    #     class_target = class_target.to(self.device, non_blocking=True)
    
    #     if isinstance(seg_target, list):
    #         seg_target = [i.to(self.device, non_blocking=True) for i in seg_target]
    #     else:
    #         seg_target = seg_target.to(self.device, non_blocking=True)
    
    #     self.optimizer.zero_grad(set_to_none=True)
    
    #     with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
    #         output = self.network(data)
    
    #         # Handle output formats (dict or tuple)
    #         if isinstance(output, dict):
    #             seg_output = output['seg']
    #             class_output = output['class']
    #             output_dict = output
    #         elif isinstance(output, (tuple, list)) and len(output) == 2:
    #             seg_output, class_output = output
    #             output_dict = {'seg': seg_output, 'class': class_output}
    #         else:
    #             raise RuntimeError("Expected output to be dict or tuple of (seg, class)")
    
    #         target_dict = {
    #             'target': seg_target,
    #             'class': class_target
    #         }
    
    #         loss = self.loss(output_dict, target_dict)
    #     return {
    #         'loss': loss.item()
            
    #     }
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        seg_target = batch['target']
        class_target = batch['class']
    
        if class_target.numel() == 0:
            raise ValueError("❌ ERROR: class_target is empty. Check your dataloader output!")
    
        data = data.to(self.device, non_blocking=True)
        class_target = class_target.to(self.device, non_blocking=True)
        seg_target = [i.to(self.device, non_blocking=True) for i in seg_target] if isinstance(seg_target, list) else seg_target.to(self.device, non_blocking=True)
    
        self.optimizer.zero_grad(set_to_none=True)
    
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
    
            if isinstance(output, dict):
                seg_output = output['seg']
                class_output = output['class']
            elif isinstance(output, (tuple, list)) and len(output) == 2:
                seg_output, class_output = output
            else:
                raise RuntimeError("Expected output to be dict or tuple of (seg, class)")
    
            target_dict = {'target': seg_target, 'class': class_target}
            loss = self.loss({'seg': seg_output, 'class': class_output}, target_dict)
    
        # Debug gradients (very helpful if loss doesn’t go down)
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in: {name}")
                if (param.grad.abs() < 1e-7).all():
                    print(f"Very small gradients in: {name}")
        
        
        # Gradient step
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
        
            # 🔍 Log unclipped raw gradient norm BEFORE clipping
            # total_grad_norm_raw = torch.norm(torch.stack([
            #     p.grad.detach().data.norm(2) for p in self.network.parameters() if p.grad is not None
            # ]), 2).item()
            # print(f"[DEBUG] Grad Norm (raw): {total_grad_norm_raw:.4f}")
        
            # Now clip
            nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        
        else:
            loss.backward()
        
            # Log raw grad norm BEFORE clipping
            # total_grad_norm_raw = torch.norm(torch.stack([
            #     p.grad.detach().data.norm(2) for p in self.network.parameters() if p.grad is not None
            # ]), 2).item()
            # print(f"[DEBUG] Grad Norm (raw): {total_grad_norm_raw:.4f}")
        
            # 💥 Clip it
            nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        # log the clipped one too (this is what you’ve always had)
        # total_grad_norm_clipped = torch.norm(torch.stack([
        #     p.grad.norm(2) for p in self.network.parameters() if p.grad is not None
        # ]), 2).item()
        # print(f"[DEBUG] Grad Norm (clipped): {total_grad_norm_clipped:.4f}")
    
        return {'loss': loss.item()}

        # # Gradient update
        # if self.grad_scaler is not None:
        #     self.grad_scaler.scale(loss).backward()
        #     self.grad_scaler.unscale_(self.optimizer)
        #     nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        #     self.grad_scaler.step(self.optimizer)
        #     self.grad_scaler.update()
        # else:
        #     loss.backward()
        #     nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        #     self.optimizer.step()
    
        
        # # Use highest resolution output if deep supervision is used
        # if isinstance(seg_output, list):
        #     seg_output = seg_output[0]
        # if isinstance(seg_target, list):
        #     seg_target = seg_target[0]
            
        # # Softmax → Argmax prediction (C, H, W, D) → (H, W, D)
        # pred_seg = seg_output.argmax(dim=1).detach().cpu().numpy()
        # true_seg = seg_target.detach().cpu().numpy()
    
        # Compute TP/FP/FN per class for Dice tracking
        # num_classes = seg_output.shape[1]
        # tp_hard = []
        # fp_hard = []
        # fn_hard = []
    
        # for cls in range(num_classes):
        #     p = (pred_seg == cls)
        #     t = (true_seg == cls)
    
        #     tp = np.sum(np.logical_and(p, t))
        #     fp = np.sum(np.logical_and(p, np.logical_not(t)))
        #     fn = np.sum(np.logical_and(np.logical_not(p), t))
    
        #     tp_hard.append(tp)
        #     fp_hard.append(fp)
        #     fn_hard.append(fn)
        # return {
        #     'loss': loss.item(),
        #     'predicted_segmentation': pred_seg,
        #     'target': true_seg,
        #     'true_class': class_target.detach().cpu().numpy(),
        #     'pred_class': class_output.argmax(1).detach().cpu().numpy(),
        #     'tp_hard': np.array(tp_hard, dtype=np.int64),
        #     'fp_hard': np.array(fp_hard, dtype=np.int64),
        #     'fn_hard': np.array(fn_hard, dtype=np.int64),
        # }


    # def on_train_epoch_end(self, train_outputs: list[dict]):
    #     outputs = collate_outputs(train_outputs)
    
    #     # ---------- Segmentation Loss ----------
    #     if self.is_ddp:
    #         losses_tr = [None for _ in range(dist.get_world_size())]
    #         dist.all_gather_object(losses_tr, outputs['loss'])
    #         loss_here = np.vstack(losses_tr).mean()
    #     else:
    #         loss_here = np.mean(outputs['loss'])
    
    #     self.logger.log('train_losses', loss_here, self.current_epoch)
    #     self.print_to_log_file(f"[Train] Seg Loss: {loss_here:.4f}")
    
    #     # ---------- Classification Metrics ----------
    #     if 'true_class' in outputs and 'pred_class' in outputs:
    #         all_true = outputs['true_class']
    #         all_pred = outputs['pred_class']
    
    #         acc = accuracy_score(all_true, all_pred)
    #         f1 = f1_score(all_true, all_pred, average='macro')
    
    #         self.logger.log('train_classification_acc', acc, self.current_epoch)
    #         self.logger.log('train_classification_f1', f1, self.current_epoch)
    #         self.print_to_log_file(f"[Train] Cls Acc: {acc:.4f} | Macro F1: {f1:.4f}")
    
    #     # ---------- Per-Class DSC from TP/FP/FN ----------
    #     if 'tp_hard' in outputs:
    #         tp = np.sum(outputs['tp_hard'], 0)
    #         fp = np.sum(outputs['fp_hard'], 0)
    #         fn = np.sum(outputs['fn_hard'], 0)
    
    #         if self.is_ddp:
    #             world_size = dist.get_world_size()
    
    #             tps = [None for _ in range(world_size)]
    #             dist.all_gather_object(tps, tp)
    #             tp = np.vstack([i[None] for i in tps]).sum(0)
    
    #             fps = [None for _ in range(world_size)]
    #             dist.all_gather_object(fps, fp)
    #             fp = np.vstack([i[None] for i in fps]).sum(0)
    
    #             fns = [None for _ in range(world_size)]
    #             dist.all_gather_object(fns, fn)
    #             fn = np.vstack([i[None] for i in fns]).sum(0)
    
    #         dice_per_class = [2 * t / (2 * t + f + n + 1e-8) for t, f, n in zip(tp, fp, fn)]
    #         mean_fg_dice = np.nanmean(dice_per_class[1:])  # skip background (label 0)
    
    #         self.logger.log('train_dice_per_class', dice_per_class, self.current_epoch)
    #         self.logger.log('train_mean_fg_dice', mean_fg_dice, self.current_epoch)
    #         self.print_to_log_file(f"[Train] Mean FG DSC: {mean_fg_dice:.4f}")
    #         self.print_to_log_file(f"[Train] DSC per class: {dice_per_class}")
    
    #     # ---------- Explicit Dice for PDF Requirement ----------
    #     if 'predicted_segmentation' in outputs and 'target' in outputs:
    #         preds = outputs['predicted_segmentation']
    #         gts = outputs['target']
    
    #         dsc_whole = []
    #         dsc_lesion = []
    
    #         for pred, gt in zip(preds, gts):
    #             pred_np = pred.astype(np.uint8)
    #             gt_np = gt.astype(np.uint8)
    
    #             try:
    #                 # Whole pancreas (label > 0)
    #                 dsc_whole.append(dc((pred_np > 0), (gt_np > 0)))
    #             except:
    #                 dsc_whole.append(0.0)
    
    #             try:
    #                 # Lesion only (label == 2)
    #                 dsc_lesion.append(dc((pred_np == 2), (gt_np == 2)))
    #             except:
    #                 dsc_lesion.append(0.0)
    
    #         mean_whole = np.mean(dsc_whole)
    #         mean_lesion = np.mean(dsc_lesion)
    
    #         self.logger.log('train_dice_whole_pancreas', mean_whole, self.current_epoch)
    #         self.logger.log('train_dice_lesion_only', mean_lesion, self.current_epoch)
    
    #         self.print_to_log_file(f"[Train] DSC Whole Pancreas (label > 0): {mean_whole:.4f}")
    #         self.print_to_log_file(f"[Train] DSC Lesion Only (label == 2): {mean_lesion:.4f}")
    
    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
    
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
    
        # Use your own logged Dice metrics
        if 'val_dice_per_class' in self.logger.my_fantastic_logging and self.logger.my_fantastic_logging['val_dice_per_class']:
            self.print_to_log_file(
                'Val Dice per class',
                [np.round(i, decimals=4) for i in self.logger.my_fantastic_logging['val_dice_per_class'][-1]]
            )
        if 'val_mean_fg_dice' in self.logger.my_fantastic_logging and self.logger.my_fantastic_logging['val_mean_fg_dice']:
            self.print_to_log_file(
                f"Val Mean FG Dice: {np.round(self.logger.my_fantastic_logging['val_mean_fg_dice'][-1], decimals=4)}"
            )
    
        # Epoch timing
        epoch_time = self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - \
                     self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]
        self.print_to_log_file(f"Epoch time: {np.round(epoch_time, decimals=2)} s")
    
        # Periodic checkpoint saving
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, f'checkpoint_latest{current_epoch + 1}.pth'))
    
        # Best checkpointing based on mean FG Dice
        if hasattr(self, "_best_val_dice") is False:
            self._best_val_dice = -1.0
        if not hasattr(self, "_best_cls_f1"):
            self._best_cls_f1 = None
    
        # -------------------------------
        #  Best SEGMENTATION checkpoint
        # -------------------------------
        current_fg_dice = self.logger.my_fantastic_logging['val_mean_fg_dice'][-1]
        if self._best_val_dice is None or current_fg_dice > self._best_val_dice:
            self._best_val_dice = current_fg_dice
            filename = f'checkpoint_best_dice_epoch{current_epoch + 1}.pth'
            self.save_checkpoint(join(self.output_folder, filename))
            self.print_to_log_file(f"New BEST VAL FG Dice: {np.round(current_fg_dice, 4)}")
        
        # -------------------------------
        #  Best CLASSIFICATION checkpoint
        # -------------------------------
        current_f1 = self.logger.my_fantastic_logging['val_classification_f1'][-1]
        if self._best_cls_f1 is None or current_f1 > self._best_cls_f1:
            self._best_cls_f1 = current_f1
            filename = f'checkpoint_best_cls_epoch{current_epoch + 1}.pth'
            self.save_checkpoint(join(self.output_folder, filename))
            self.print_to_log_file(f"New BEST CLS F1: {np.round(current_f1, 4)}")

    
        # Plot if main process
        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)
    
        self.current_epoch += 1


    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        seg_target = batch['target']
        class_target = batch['class']  # classification label
    
        data = data.to(self.device, non_blocking=True)
        class_target = class_target.to(self.device, non_blocking=True)
    
        if isinstance(seg_target, list):
            seg_target = [i.to(self.device, non_blocking=True) for i in seg_target]
        else:
            seg_target = seg_target.to(self.device, non_blocking=True)
    
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            if isinstance(output, dict):
                seg_output = output['seg']
                class_output = output['class']
            else:
                seg_output, class_output = output
    
            target_dict = {
                'target': seg_target,
                'class': class_target
            }
            l = self.loss({'seg': seg_output, 'class': class_output}, target_dict)
    
        if self.enable_deep_supervision:
            seg_output = seg_output[0]
            seg_target = seg_target[0]
    
        # Segmentation prediction
        axes = [0] + list(range(2, seg_output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
        else:
            output_seg = seg_output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(seg_output.shape, device=seg_output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg
    
        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (seg_target != self.label_manager.ignore_label).float()
                seg_target[seg_target == self.label_manager.ignore_label] = 0
            else:
                if seg_target.dtype == torch.bool:
                    mask = ~seg_target[:, -1:]
                else:
                    mask = 1 - seg_target[:, -1:]
                seg_target = seg_target[:, :-1]
        else:
            mask = None
    
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, seg_target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
    
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
    
        # Classification predictions
        pred_class = class_output.argmax(dim=1).detach().cpu().numpy()
        true_class = class_target.detach().cpu().numpy()
    
        # convert the argmax prediction to match shape (B, H, W, D)
        predicted_seg = seg_output.argmax(dim=1).detach().cpu().numpy()
        true_seg = seg_target.detach().cpu().numpy()
        
        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'true_class': true_class,       # for macro-F1
            'pred_class': pred_class,        # for macro-F1
            'predicted_segmentation': predicted_seg,  # For DSC (whole / lesion)
            'target': true_seg                         # For DSC (whole / lesion)
        }

        
    def on_validation_epoch_end(self, val_outputs: list[dict]):
        outputs = collate_outputs(val_outputs)
    
        # ---------- Segmentation Loss ----------
        if self.is_ddp:
            losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_val, outputs['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs['loss'])
    
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.print_to_log_file(f"[Val] Seg Loss: {loss_here:.4f}")
    
        # ---------- Classification Metrics ----------
        if 'true_class' in outputs and 'pred_class' in outputs:
            all_true = np.asarray(outputs['true_class']).reshape(-1)  # ensure shape (N,)
            all_pred = np.asarray(outputs['pred_class']).reshape(-1)  # ensure shape (N,)
    
            acc = accuracy_score(all_true, all_pred)
            f1 = f1_score(all_true, all_pred, average='macro')
    
            self.logger.log('val_classification_acc', acc, self.current_epoch)
            self.logger.log('val_classification_f1', f1, self.current_epoch)
            self.print_to_log_file(f"[Val] Cls Acc: {acc:.4f} | Macro F1: {f1:.4f}")
            
    
        # ---------- Segmentation Metrics via TP/FP/FN ----------
        if 'tp_hard' in outputs:
            tp = np.sum(outputs['tp_hard'], 0)
            fp = np.sum(outputs['fp_hard'], 0)
            fn = np.sum(outputs['fn_hard'], 0)
    
            if self.is_ddp:
                world_size = dist.get_world_size()
    
                tps = [None for _ in range(world_size)]
                dist.all_gather_object(tps, tp)
                tp = np.vstack([i[None] for i in tps]).sum(0)
    
                fps = [None for _ in range(world_size)]
                dist.all_gather_object(fps, fp)
                fp = np.vstack([i[None] for i in fps]).sum(0)
    
                fns = [None for _ in range(world_size)]
                dist.all_gather_object(fns, fn)
                fn = np.vstack([i[None] for i in fns]).sum(0)
    
            dice_per_class = [2 * t / (2 * t + f + n + 1e-8) for t, f, n in zip(tp, fp, fn)]
            mean_fg_dice = np.nanmean(dice_per_class[1:])  # skip background
            self.logger.log('val_dice_per_class', dice_per_class, self.current_epoch)
            self.logger.log('val_mean_fg_dice', mean_fg_dice, self.current_epoch)
            self.print_to_log_file(f"[Val] Mean FG DSC: {mean_fg_dice:.4f}")
            self.print_to_log_file(f"[Val] DSC per class: {dice_per_class}")
            
            dsc_whole = []
            dsc_lesion = []
            
            for pred, gt in zip(outputs['predicted_segmentation'], outputs['target']):
                pred = np.asarray(pred).astype(np.uint8)
                gt = np.asarray(gt).astype(np.uint8)
            
                try:
                    dsc_whole.append(dc((pred > 0), (gt > 0)))  # DSC of all foreground (pancreas + lesion)
                except:
                    dsc_whole.append(0.0)
            
                try:
                    dsc_lesion.append(dc((pred == 2), (gt == 2)))  # DSC of lesion only
                except:
                    dsc_lesion.append(0.0)
            
            mean_whole = np.mean(dsc_whole)
            mean_lesion = np.mean(dsc_lesion)
            
            # Log them
            self.logger.log('val_dice_whole_pancreas', mean_whole, self.current_epoch)
            self.logger.log('val_dice_lesion_only', mean_lesion, self.current_epoch)

            # Print to log file
            self.print_to_log_file(f"[Val] DSC Whole Pancreas (label > 0): {mean_whole:.4f}")
            self.print_to_log_file(f"[Val] DSC Lesion Only (label == 2): {mean_lesion:.4f}")
    
            
        # Early stopping logic
        if loss_here < self.best_val_loss - 1e-5:  # small improvement threshold
            self.best_val_loss = loss_here
            self.epochs_without_improvement = 0
        
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))
            self.print_to_log_file(f"New best val loss: {loss_here:.4f} (saved best model)")
        
        else:
            self.epochs_without_improvement += 1
            self.print_to_log_file(f"No improvement. Patience counter: {self.epochs_without_improvement}/{self.patience}")
        
        # Trigger early stop only after minimum number of epochs
        if self.current_epoch >= self.min_epochs and self.epochs_without_improvement >= self.patience:
            self.print_to_log_file(f"Early stopping triggered at epoch {self.current_epoch } Minimum epoch reached.")
            self._stop_training = True


    

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()
    
        predictor = nnUNetPredictorWithClassification(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        predictor.manual_initialization(
            self.network,
            self.plans_manager,
            self.configuration_manager,
            parameters=[self.network.state_dict()],
            dataset_json=self.dataset_json,
            trainer_name=self.__class__.__name__,
            inference_allowed_mirroring_axes=self.inference_allowed_mirroring_axes
        )
    
        validation_output_folder = join(self.output_folder, 'validation')
        maybe_mkdir_p(validation_output_folder)
    
        _, val_keys = self.do_split()
        if self.is_ddp:
            last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
            val_keys = val_keys[self.local_rank::dist.get_world_size()]
    
        dataset_val = self.dataset_class(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
    
        # Setup preprocessing pipeline
        data_iterator = predictor.get_data_iterator_from_raw_npy_data(
            image_or_list_of_images=[
                np.asarray(dataset_val.load_case(k)[0]) for k in dataset_val.identifiers
            ],
            segs_from_prev_stage_or_list_of_segs_from_prev_stage=[
                np.asarray(dataset_val.load_case(k)[2]) if dataset_val.load_case(k)[2] is not None else None
                for k in dataset_val.identifiers
            ],
            properties_or_list_of_properties=[
                dataset_val.load_case(k)[3] for k in dataset_val.identifiers
            ],
            truncated_ofname=[
                join(validation_output_folder, k) for k in dataset_val.identifiers
            ]
        )
    
        # Safe segmentation override
        original_network = predictor.network
        predictor.network = lambda x: original_network(x)[0]
    
        predictor.predict_from_data_iterator(
            data_iterator,
            save_probabilities=save_probabilities,
            num_processes_segmentation_export=default_num_processes
        )
    
        predictor.network = original_network  # restore full model
    
        if self.is_ddp:
            dist.barrier()
    
        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(
                join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                validation_output_folder,
                join(validation_output_folder, 'summary.json'),
                self.plans_manager.image_reader_writer_class(),
                self.dataset_json["file_ending"],
                self.label_manager.foreground_regions if self.label_manager.has_regions else self.label_manager.foreground_labels,
                self.label_manager.ignore_label,
                chill=True,
                num_processes=default_num_processes * dist.get_world_size() if self.is_ddp else default_num_processes
            )
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice:", metrics['foreground_mean']["Dice"], also_print_to_console=True)
    
            subtype_csv = os.path.join(validation_output_folder, "subtype_results.csv")
            if os.path.exists(subtype_csv):
                df_preds = pd.read_csv(subtype_csv)
                true_labels = [dataset_val.class_label_map.get(k, -1) for k in dataset_val.identifiers]
                pred_labels = [df_preds[df_preds["Names"] == k + ".nii.gz"]["Subtype"].values[0] for k in dataset_val.identifiers]
    
                acc = accuracy_score(true_labels, pred_labels)
                f1 = f1_score(true_labels, pred_labels, average='macro')
                self.print_to_log_file(f"[Val] Cls Acc: {acc:.4f} | Macro F1: {f1:.4f}", also_print_to_console=True)
    
                try:
                    import wandb
                    wandb.log({
                        "val/classification_accuracy": acc,
                        "val/classification_macro_f1": f1,
                        "val/dice": metrics['foreground_mean']["Dice"]
                    }, step=self.epoch)
                except ImportError:
                    self.print_to_log_file("wandb not installed. Skipping W&B logging.")
            else:
                self.print_to_log_file("subtype_results.csv not found. Skipping classification evaluation.", also_print_to_console=True)
    
        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
    
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod
    
                # Extract epoch number from filename (e.g. "_epoch34")
                match = re.search(r"_epoch(\d+)", filename)
                epoch_from_filename = int(match.group(1)) if match else None
    
                # Create cleaned filename (remove "_epochXX")
                filename_base = os.path.basename(filename)
                filename_base_cleaned = re.sub(r"_epoch\d+", "", filename_base)
                cleaned_filename = os.path.join(os.path.dirname(filename), filename_base_cleaned)
    
                # Save the checkpoint to cleaned filename
                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
    
                torch.save(checkpoint, cleaned_filename)
                self.print_to_log_file(f"Saved checkpoint to {cleaned_filename}")
    
                # Log to wandb
                if wandb.run is not None:
                    artifact_name = os.path.splitext(filename_base_cleaned)[0]  # remove .pth
                    artifact = wandb.Artifact(artifact_name, type="model")
    
                    # Add epoch to artifact metadata
                    if epoch_from_filename is not None:
                        artifact.metadata = {"epoch": epoch_from_filename}
                        wandb.config.update({"logged_epoch": epoch_from_filename}, allow_val_change=True)
    
                    artifact.add_file(cleaned_filename, name=filename_base_cleaned)
                    wandb.log_artifact(artifact)
    
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def run_training(self):
        self.on_train_start()

        if self.local_rank == 0 and wandb.run is None:
            wandb.init(
                project="pancreas_multitask",
                save_code=False,
                config={
                    "trainer": self.__class__.__name__,
                    "architecture": "nnUNetv2_with_classification_head",
                    "dataset": "Dataset050_MyProject",
                    "configuration": "3d_fullres",
                    "Augmentation":"No",
                    "description": "UNet model with classification head using first decoder output (C=320). GlobalAvgPool + FC used to predict subtype. Balances semantic abstraction and task-specific decoding features."
                }
            )
        
            
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            # Check for early stopping flag
            if hasattr(self, '_stop_training') and self._stop_training:
                self.print_to_log_file("Training stopped early due to no validation improvement.")
                break

        self.on_train_end()
        if wandb.run is not None:
            wandb.finish()
    
