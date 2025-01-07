import torch
import model
from pathlib import Path
import os
import numpy as np
import json
from get_augmentor import Augmentor


def save_checkpoint(
    checkpoint_path, model, optimizer, scheduler, tokenizer, config, results
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.parent.exists():
        checkpoint_path.parent.mkdir(parents=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "tokenizer": tokenizer,
            "config": config,
            "results": results,
        },
        Path(checkpoint_path),
    )


def load_checkpoint(checkpoint_path, weight_only=False):
    checkpoint = torch.load(Path(checkpoint_path), weights_only=weight_only)

    config = checkpoint["config"]

    gpt = model.GPT(config=config).to(config.device)

    gpt.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(gpt.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    tokenizer = checkpoint["tokenizer"]

    results = checkpoint["results"]

    return_dict = {
        "model": gpt,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "tokenizer": tokenizer,
        "config": config,
        "results": results,
    }

    return return_dict

def create_data(config, tokenizer, is_train, save_folder="data/pretraining"):
    data_path = Path(config.data_path)
    filelist = os.listdir(data_path)
    augmentor = Augmentor(config.vocab_size, tokenizer.special_tokens)
    data = []
    for file in filelist:
        json_path = data_path / file
        with open(json_path, "r") as fhandle:
            task = json.load(fhandle)
        if is_train:
            task = task["train"]
            augmentor.apply(task)  # In-place change
        else:
            task = task["test"]
        task = tokenizer.encode(task)
        np_task = np.array(task, dtype=np.uint8)
        data.append(np_task)
    data = np.concatenate(data)
    save_folder = Path(save_folder)
    if not save_folder.exists():
        save_folder.mkdir(parents=True)
    if is_train:
        np.save(save_folder / "training.npy", data)
    else:
        np.save(save_folder / "validation.npy", data)