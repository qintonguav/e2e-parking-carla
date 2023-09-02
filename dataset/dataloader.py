import torch
import numpy as np
import random
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from dataset.carla_dataset import CarlaDataset
from tool.config import Configuration


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ParkingDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.data_dir = self.cfg.data_dir
        self.train_loader = None
        self.val_loader = None

    def setup(self, stage: str):
        data_root = self.data_dir
        train_set = CarlaDataset(data_root, 1, self.cfg)
        val_set = CarlaDataset(data_root, 0, self.cfg)
        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=self.cfg.batch_size,
                                       shuffle=True,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=True)
        self.val_loader = DataLoader(dataset=val_set,
                                     batch_size=self.cfg.batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     pin_memory=True,
                                     drop_last=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
