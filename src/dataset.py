"""
Load the per-client feature partitions written by prepare_data.py.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config


class FeatureDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _load(data_dir, cid, split):
    x = np.load(os.path.join(data_dir, f"client_{cid}_x_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"client_{cid}_y_{split}.npy"))
    return x, y


def get_loaders(cid, data_dir="client_data", batch_size=config.BATCH_SIZE):
    x_tr, y_tr = _load(data_dir, cid, "train")
    x_va, y_va = _load(data_dir, cid, "val")
    train = DataLoader(FeatureDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    val = DataLoader(FeatureDataset(x_va, y_va), batch_size=batch_size, shuffle=False)
    return train, val, len(x_tr)
