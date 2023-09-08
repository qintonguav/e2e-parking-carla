import os
import sys
import argparse
import yaml

from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from trainer.pl_trainer import ParkingTrainingModule, setup_callbacks
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.dataloader import ParkingDataModule
from tool.config import get_cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train():
    arg_parser = argparse.ArgumentParser(description='ParkingModel')
    arg_parser.add_argument(
        '--config',
        default='./config/training.yaml',
        type=str,
        help='path to training.yaml (default: ./config/training.yaml)')
    args = arg_parser.parse_args()

    with open(args.config, 'r') as yaml_file:
        try:
            cfg_yaml = yaml.safe_load(yaml_file)
        except yaml.YAMLError:
            logger.exception("Open {} failed!", args.config)
    cfg = get_cfg(cfg_yaml)

    logger.remove()
    logger.add(cfg.log_dir + '/training_{time}.log', enqueue=True, backtrace=True, diagnose=True)
    logger.add(sys.stderr, enqueue=True)
    logger.info("Config Yaml File: {}", args.config)

    seed_everything(42)

    parking_callbacks = setup_callbacks(cfg)
    tensor_logger = TensorBoardLogger(save_dir=cfg.log_dir, default_hp_metric=False)
    num_gpus = 1

    parking_trainer = Trainer(callbacks=parking_callbacks,
                              logger=tensor_logger,
                              accelerator='gpu',
                              strategy='ddp' if num_gpus > 1 else None,
                              devices=num_gpus,
                              max_epochs=cfg.epochs,
                              log_every_n_steps=cfg.log_every_n_steps,
                              check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                              profiler='simple')

    parking_model = ParkingTrainingModule(cfg)
    parking_datamodule = ParkingDataModule(cfg)
    parking_trainer.fit(parking_model, datamodule=parking_datamodule)


if __name__ == '__main__':
    train()
