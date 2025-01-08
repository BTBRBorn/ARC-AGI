from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data_path, block_size):
        #mmap_mode has to be "r" without this augmentors color changes
        #would not be applied.
        self.data = np.load(Path(data_path), mmap_mode="r")
        self.block_size = block_size 

    def __len__(self):
        if len(self.data) % self.block_size:
            return (len(self.data) // self.block_size) + 1
        else:
            return len(self.data) // self.block_size

    def __getitem__(self, index):
        buff = self.data[index * self.block_size : (index + 1) * self.block_size + 1]
        if len(buff) != self.block_size + 1:
            buff = self.data[-self.block_size - 1:]
        buff = torch.tensor(buff, dtype=torch.long)
        x, y = buff[:-1], buff[1:]
        return x, y


def create_dataloaders(config, train_shuffle=True):
    train_dataset = CustomDataset("data/pretraining/training.npy", config.block_size)
    val_dataset = CustomDataset("data/pretraining/validation.npy", config.block_size)
    train_dataloader = DataLoader(
        train_dataset,
        config.batch_size,
        shuffle=train_shuffle,
        num_workers=config.dataloader_num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        config.batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
    )
    return train_dataloader, val_dataloader
