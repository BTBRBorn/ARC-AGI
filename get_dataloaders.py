from torch.utils.data import Dataset, DataLoader
import os
import json
from pathlib import Path
import torch
import random


def flatten(array):
    flat_l = []
    for e in array:
        flat_l.extend(e)
    return flat_l

def change_colors(task:list[list], colors:list):

    def get_mappings(example, colors):
        flattened = flatten(example['input']) + flatten(example['output'])
        color_set = set(flattened).difference({0})
        mappings = {}
        copy_colors = list(colors)
        for c in color_set:
            new_c = random.choice(copy_colors)
            mappings[c] = new_c
            copy_colors.remove(new_c)
        return mappings

    def change_array(array, mappings):
        n_rows, n_columns = len(array), len(array[0])
        for i in range(n_rows):
            for j in range(n_columns):
                if array[i][j] != 0:
                    array[i][j] = mappings[array[i][j]]

    def change_one_example(example, colors):
        mappings = get_mappings(example, colors)
        change_array(example['input'], mappings)
        change_array(example['output'], mappings)
    
    for example in task:
        change_one_example(example, colors)


def add_special_tokens(array, special_tokens, block_size):

    if special_tokens is not None:
        data = flatten(
            [
                [special_tokens["start_of_input"]]
                + flatten(e["input"])
                + [special_tokens["end_of_input"]]
                + [special_tokens["start_of_output"]]
                + flatten(e["output"])
                + [special_tokens["end_of_output"]]
                for e in array
            ]
        )
        assert len(data) <= block_size, f"Data length ({len(data)}) can't be bigger than block_size ({block_size})"
        # + 1 is needed because buffer size needs to be block_size + 1
        data = data + [special_tokens['fill_value']]*(block_size-len(data) + 1) 
    else:
        data = flatten([flatten(e["input"]) + flatten(e["output"]) for e in array])


    return data


transforms = {"change_colors": change_colors, "add_special_tokens": add_special_tokens}


class CustomDataset(Dataset):
    def __init__(
        self, config, is_train=True, transforms=transforms, 
    ):

        self.data_path = Path(config.data_path)
        self.block_size = config.block_size if is_train else config.test_block_size
        self.files = os.listdir(config.data_path)
        self.special_tokens = {
            "start_of_input": None,
            "end_of_input": None,
            "start_of_output": None,
            "end_of_output": None,
            "fill_value": None,
        }
        last_token = config.vocab_size - len(self.special_tokens)
        for key in self.special_tokens.keys():
            self.special_tokens[key] = last_token
            last_token += 1

        self.transforms = transforms
        self.is_train = is_train
        self.colors = list(set(range(1, config.vocab_size)) - set(self.special_tokens.values()))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        json_path = self.data_path / file
        with open(json_path, "r") as fhandle:
            task = json.load(fhandle)
        #Transformations are done in place
        if self.is_train:
            task = task["train"]
            self.transforms["change_colors"](task, self.colors)
            task = self.transforms["add_special_tokens"](task, self.special_tokens, self.block_size)
        else:
            task = task["test"]
            task = self.transforms["add_special_tokens"](task, self.special_tokens, self.block_size)

        buff = torch.tensor(task, dtype=torch.long)
        x, y = buff[:-1], buff[1:]
        return x, y


"""
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
            buffer = self.data[-self.buffer_size :]

        buffer = torch.tensor(buffer, dtype=torch.long)
        x, y = buffer[:-1], buffer[1:]

        return x, y
"""


def create_dataloaders(config):
    train_dataset = CustomDataset(config)
    val_dataset = CustomDataset(config, is_train=False)
    train_dataloader = DataLoader(
        train_dataset, config.batch_size, shuffle=True, num_workers=config.dl_num_workers
    )
    val_dataloader = DataLoader(val_dataset, config.batch_size, num_workers=config.dl_num_workers)
    return train_dataloader, val_dataloader
