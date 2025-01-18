import torch
import model
from pathlib import Path
import os
import numpy as np
import json
from get_augmentor import Augmentor
import random
import matplotlib.pyplot as plt


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

    optim_groups = [
        {
            "params": [param for param in gpt.parameters() if param.dim() >= 2],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [param for param in gpt.parameters() if param.dim() < 2],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, fused=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.scheduler_iter
    )
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


def plot_losses(results):
    train_loss, val_loss = results["train_losses"], results["val_losses"]
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="training", color="green")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="validation", color="red")
    plt.title("Train and Validation Losses vs Num Epochs")
    plt.xlabel("Num Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
