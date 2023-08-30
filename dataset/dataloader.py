import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

