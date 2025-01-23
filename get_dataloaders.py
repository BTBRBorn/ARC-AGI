from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import numpy as np
from get_augmentor import Augmentor
import os
import json
import random


def create_data(
    data_path,
    vocab_size,
    tokenizer,
    save_folder="pretraining/",
    is_train=True,
    rolled=True,
    augmented=True,
):
    data_path = Path(data_path)
    filelist = os.listdir(data_path)
    augmentor = Augmentor(vocab_size, tokenizer.special_tokens)
    data = []
    for file in filelist:
        json_path = data_path / file
        with open(json_path, "r") as fhandle:
            task = json.load(fhandle)
        if is_train:
            task = task["train"]
        else:
            task = task["test"]
        if augmented:
            augmentor.apply(task)  # In-place change
        task = tokenizer.encode(task)
        np_task = np.array(task, dtype=np.uint8)
        data.append(np_task)

    data = np.concatenate(data)

    save_folder = data_path.parent / save_folder
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    if rolled:
        data = np.roll(data, shift=random.randint(0, 50000))

    if is_train:
        np.save(save_folder / "training.npy", data)
    else:
        np.save(save_folder / "validation.npy", data)


class CustomDataset(Dataset):
    def __init__(self, data_path, block_size):
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
            buff = self.data[-self.block_size - 1 :]
        buff = torch.tensor(buff, dtype=torch.long)
        x, y = buff[:-1], buff[1:]
        return x, y


def create_dataloaders(config, tokenizer, train_shuffle=True):
    train_dataset_path = config.data_path_train.parent / "pretraining/training.npy"
    val_dataset_path = config.data_path_val.parent / "pretraining/validation.npy"
    # Apply changes to the training dataset
    create_data(
        data_path=config.data_path_train,
        vocab_size=config.vocab_size,
        tokenizer=tokenizer,
        save_folder="pretraining/",
        is_train=True,
        rolled=True,
        augmented=True,
    )

    train_dataset = CustomDataset(train_dataset_path, config.block_size)
    val_dataset = CustomDataset(val_dataset_path, config.block_size)
    
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
