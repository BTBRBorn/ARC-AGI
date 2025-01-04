from torch.utils.data import Dataset, DataLoader
import os
import json
from pathlib import Path
import torch
from get_augmentor import Augmentor

class CustomDataset(Dataset):
    def __init__(self, config, tokenizer, is_train):
        self.data_path = Path(config.data_path)
        self.files = os.listdir(config.data_path)
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.augmentor = Augmentor(config.vocab_size, tokenizer.special_tokens)
        self.block_size = config.block_size if is_train else config.test_block_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        json_path = self.data_path / file
        with open(json_path, "r") as fhandle:
            task = json.load(fhandle)
        if self.is_train:
            task = task['train']
            self.augmentor.apply(task) #In-place change
        else:
            task = task['test'] 

        task = self.tokenizer.encode(task, self.block_size)
        buff = torch.tensor(task, dtype=torch.long)
        x, y = buff[:-1], buff[1:]
        return x, y


def create_dataloaders(config, tokenizer):
    train_dataset = CustomDataset(config, tokenizer, is_train=True)
    val_dataset = CustomDataset(config, tokenizer, is_train=False)
    train_dataloader = DataLoader(
        train_dataset,
        config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        config.batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
    )
    return train_dataloader, val_dataloader
