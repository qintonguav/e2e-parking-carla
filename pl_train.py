import argparse
import os
import sys
from loguru import logger
import yaml
from tool.config import get_cfg
from pytorch_lightning import Trainer
from trainer.pl_trainer import ParkingTrainingModule, setup_callbacks
from pytorch_lightning.loggers import TensorBoardLogger

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
        cfg_yaml = yaml.safe_load(yaml_file)
    cfg = get_cfg(cfg_yaml)

    logger.remove()
    logger.add(cfg.log_dir + '/training_{time}.log', enqueue=True, backtrace=True, diagnose=True)
    logger.add(sys.stderr, enqueue=True)

    parking_callbacks = setup_callbacks(cfg)
    tensor_logger = TensorBoardLogger(save_dir=cfg.log_dir)
    num_gpus = 8

    # set dataloader

    parking_model = ParkingTrainingModule(cfg)
    parking_trainer = Trainer(callbacks=parking_callbacks,
                              logger=tensor_logger,
                              gpus=num_gpus)
    parking_trainer.fit(parking_model)


if __name__ == '__main__':
    train()
