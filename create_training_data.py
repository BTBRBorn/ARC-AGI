import argparse
from pathlib import Path
from get_tokenizer import Tokenizer
from tqdm.auto import tqdm
import pickle
import shutil
import numpy as np
from get_augmentor import Augmentor
import os
import random
import json
from get_arc_generator import ArcGenerator

parser = argparse.ArgumentParser()

parser.add_argument("--train_data_path", type=str, default="data/finetune")
parser.add_argument("--val_data_path", type=str, default="data/finetune")
parser.add_argument("--processed_data_path", type=str, default="data/pretraining")
parser.add_argument("--num_shards", type=int, default=10)
parser.add_argument("--vocab_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=10)
parser.add_argument(
    "--tokenizer_save_path", type=str, default=""
)
parser.add_argument("--noise_epoch", type=int, default=5000)

args = parser.parse_args()


def create_data(
    source_path,
    tokenizer,
    output_file_path,
    is_train=True,
    rolled=True,
    augmented=True,
    add_noise=False,
):
    data_path = Path(source_path)
    output_file_path = Path(output_file_path)
    filelist = os.listdir(data_path)
    augmentor = Augmentor()
    
    #Add data from actual arc tasks
    tasks = []
    for file in filelist:
        json_path = data_path / file
        with open(json_path, "r") as fhandle:
            task = json.load(fhandle)
        tasks.append(task)

    if is_train: 
        #Add data from generated arc tasks
        arc_generator = ArcGenerator()
        generated_tasks = arc_generator(num_repeats=10)
        #Combined the two
        tasks.extend(generated_tasks)
    
    data = []
    for task in tasks:
        if is_train:
            task = task["train"]
        else:
            task = task["test"]
        if augmented:
            augmentor(task, add_noise=add_noise)  # In-place change
        task = tokenizer.encode(task)
        np_task = np.array(task, dtype=np.uint8)
        data.append(np_task)

    data = np.concatenate(data)

    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True)

    if rolled:
        data = np.roll(data, shift=random.randint(0, 50000))

    np.save(output_file_path, data)


if __name__ == "__main__":
    tokenizer = Tokenizer(args.vocab_size)

    PROCESSED_DATA_PATH = Path(args.processed_data_path)
    TRAIN_DATA_PATH = Path(args.train_data_path)
    VAL_DATA_PATH = Path(args.val_data_path)
    if PROCESSED_DATA_PATH.exists():
        shutil.rmtree(PROCESSED_DATA_PATH)

    print("Training data is being created.")
    for i in tqdm(range(1, args.num_shards + 1)):

        if i > args.noise_epoch:
            add_noise = bool(i % 2)
        else:
            add_noise = False

        create_data(
            source_path=TRAIN_DATA_PATH,
            tokenizer=tokenizer,
            output_file_path=PROCESSED_DATA_PATH / f"training_{i}.npy",
            is_train=True,
            rolled=True,
            augmented=True,
            add_noise=add_noise,
        )

    # Validation data is being created
    create_data(
        source_path=VAL_DATA_PATH,
        tokenizer=tokenizer,
        output_file_path=PROCESSED_DATA_PATH / "validation.npy",
        is_train=False,
        rolled=False,
        augmented=False,
    )

    if args.tokenizer_save_path:
        tokenizer.train(
            data_path=PROCESSED_DATA_PATH,
            num_workers=args.num_workers,
        )

        tokenizer_save_path = Path(args.tokenizer_save_path)
        if not tokenizer_save_path.parent.exists():
            tokenizer_save_path.parent.mkdir(parents=True)

        with open(tokenizer_save_path, "wb") as fhandle:
            pickle.dump(tokenizer, fhandle)
