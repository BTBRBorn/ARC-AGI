from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import torch


class CustomDataset(Dataset):
    def __init__(self, data_path: str | Path, block_size):
        self.data = np.load(data_path, mmap_mode="r")
        self.buffer_size = block_size + 1

    def __len__(self):
        if len(self.data) % self.buffer_size:
            return (len(self.data) // self.buffer_size) + 1
        else:
            return len(self.data) // self.buffer_size

    def __getitem__(self, idx):
        buffer = self.data[idx * self.buffer_size : (idx + 1) * self.buffer_size]
        if len(buffer) != self.buffer_size:
            buffer = self.data[-self.buffer_size:]

        buffer = torch.tensor(buffer, dtype=torch.long)
        x, y = buffer[:-1], buffer[1:]

        return x, y


def create_dataloaders(train_path, val_path, batch_size, block_size, num_workers):
    train_dataset = CustomDataset(Path(train_path), block_size)
    val_dataset = CustomDataset(Path(val_path), block_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(val_dataset, batch_size, num_workers=num_workers)
    return train_dataloader, val_dataloader
