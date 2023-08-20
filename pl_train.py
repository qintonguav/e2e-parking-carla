import argparse
import logging
import yaml
from tool.config import get_cfg
from pytorch_lightning import Trainer
from trainer.pl_trainer import ParkingTrainingModule
from pytorch_lightning.callbacks import ModelCheckpoint


def setup_callback():
    ckpt_callback = ModelCheckpoint(filepath='./checkpoints', monitor='val_loss', save_top_k=3, mode='min')
    return [ckpt_callback]


def train():
    arg_parser = argparse.ArgumentParser(description='ParkingModel')
    arg_parser.add_argument(
        '--config',
        default='./config/training.yaml',
        help='path to training.yaml (default: ./config/training.yaml)')
    args = arg_parser.parse_args()

    with open(args.config, 'r') as yaml_file:
        cfg_yaml = yaml.safe_load(yaml_file)
    cfg = get_cfg(cfg_yaml)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    parking_callback = setup_callback()

    # set dataloader

    parking_model = ParkingTrainingModule(cfg)
    parking_trainer = Trainer(callbacks=parking_callback)
    parking_trainer.fit(parking_model)


if __name__ == '__main__':
    train()
