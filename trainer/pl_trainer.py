import torch
import torch.nn as nn
import pytorch_lightning as pl
from tool.config import Configuration
from loss.control_loss import ControlLoss, ControlValLoss
from loss.depth_loss import DepthLoss
from loss.seg_loss import SegmentationLoss
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from model.parking_model import ParkingModel

def setup_callbacks(cfg):
    callbacks = []

    ckpt_callback = ModelCheckpoint(filepath=cfg.ckeckpoint_dir,
                                    monitor='val_loss',
                                    save_top_k=3,
                                    mode='min',
                                    filename='E2E-APA-{epoch:02d}-{val_loss:.2f}',
                                    save_last=True)
    callbacks.append(ckpt_callback)

    progress_bar = TQDMProgressBar()
    callbacks.append(progress_bar)

    lr_monitor = LearningRateMonitor()
    callbacks.append(lr_monitor)
    return callbacks


class ParkingTrainingModule(pl.LightningModule):
    def __init__(self, cfg: Configuration):
        super(ParkingTrainingModule, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.control_loss_func = ControlLoss(self.cfg)

        self.control_val_loss_func = ControlValLoss(self.cfg)

        self.segmentation_loss_func = SegmentationLoss(
            class_weights=torch.Tensor(self.cfg.seg_vehicle_weights)
        )

        self.depth_loss_func = DepthLoss(self.cfg)

        self.parking_model = ParkingModel(self.cfg)

    def training_step(self, batch, batch_idx):
        loss_dict = {}
        pred_control, pred_segmentation, pred_depth = self.parking_model(batch)

        control_loss = self.control_loss_func(pred_control, batch)
        loss_dict.update({
            "control_loss": control_loss
        })

        segmentation_loss = self.segmentation_loss_func(pred_segmentation, batch['segmentation'])
        loss_dict.update({
            "segmentation_loss": segmentation_loss
        })

        depth_loss = self.depth_loss_func(pred_depth, batch['depth'])
        loss_dict.update({
            "depth_loss": depth_loss
        })

        train_loss = sum(loss_dict.values())
        loss_dict.update({
            "train_loss": train_loss
        })

        self.log_segmentation(pred_segmentation, batch, 'segmentation')
        self.log_depth(pred_depth, batch, 'depth')

        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss_dict = {}
        pred_control, pred_segmentation, pred_depth = self.parking_model(batch)

        acc_steer_val_loss, reverse_val_loss = self.control_val_loss_func(pred_control, batch)
        val_loss_dict.update({
            "acc_steer_val_loss": acc_steer_val_loss,
            "reverse_val_loss": reverse_val_loss
        })

        segmentation_val_loss = self.segmentation_loss_func(pred_segmentation, batch['segmentation'])
        val_loss_dict.update({
            "segmentation_val_loss": segmentation_val_loss
        })

        depth_val_loss = self.depth_loss_func(pred_depth, batch['depth'])
        val_loss_dict.update({
            "depth_val_loss": depth_val_loss
        })

        val_loss = sum(val_loss_dict.values())
        val_loss_dict.update({
            "val_loss": val_loss
        })

        self.log_segmentation(pred_segmentation, batch, 'segmentation_val')
        self.log_depth(pred_depth, batch, 'depth_val')

        return val_loss

    def configure_optimizers(self):
        pass

    def log_segmentation(self, pred_segmentation, gt_segmentation, name):
        pass

    def log_depth(self, pred_depth, gt_depth, name):
        pass
