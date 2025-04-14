import argparse
from pathlib import Path
import random
import json
import pickle
import shutil
import numpy as np
import functools
from concurrent import futures
from copy import deepcopy
import os

from get_tokenizer import Tokenizer
from get_augmentor import Augmentor


def get_tasks(data_path):
    tasks = []
    for cur_dir_path, sub_dirs, sub_files in os.walk(data_path):
        for file in sub_files:
            if ".json" in file:
                file_path = cur_dir_path / file
            else:
                continue
            json_obj = json.loads(file_path.read_text())
            if "train" in json_obj.keys():
                tasks.append(json_obj)
            else:
                for task in json_obj.values():
                    if isinstance(task, dict) and "train" in task.keys():
                        tasks.append(task)
    return tasks


def create_data(
    output_file_path,
    source_path,
    tokenizer,
    is_train,
    rolled,
    augmented,
    num_repeat,
):
    data_path = Path(source_path)
    output_file_path = Path(output_file_path)
    augmentor = Augmentor()

    data = []
    tasks = get_tasks(data_path)
    for _ in range(num_repeat):
        copy_tasks = deepcopy(tasks)
        for task in copy_tasks:
            if is_train:
                task = task["train"]
                random.shuffle(task)
            else:
                task = task["test"]
            if augmented:
                augmentor(task)  # In-place change
            task = tokenizer.encode(task)
            np_task = np.array(task, dtype=np.uint8)
            data.append(np_task)

    if is_train:
        random.shuffle(data)
    data = np.concatenate(data)

    if rolled:
        data = np.roll(data, shift=random.randint(0, 50000))

    np.save(output_file_path, data)

    return output_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path", type=str, default="data/training")
    parser.add_argument("--val_data_path", type=str, default="data/training")
    parser.add_argument("--processed_data_path", type=str, default="data/pretraining")
    parser.add_argument("--num_shards", type=int, default=10)
    parser.add_argument("--num_repeat_per_shard", type=int, default=1)
    parser.add_argument("--vocab_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--tokenizer_save_path", type=str, default="")
    parser.add_argument("--only_validation_data", type=int, choices={0, 1}, default=0)

    args = parser.parse_args()
    tokenizer = Tokenizer(args.vocab_size)

    PROCESSED_DATA_PATH = Path(args.processed_data_path)
    TRAIN_DATA_PATH = Path(args.train_data_path)
    VAL_DATA_PATH = Path(args.val_data_path)

    if not args.only_validation_data:
        if PROCESSED_DATA_PATH.exists():
            shutil.rmtree(PROCESSED_DATA_PATH)
            PROCESSED_DATA_PATH.mkdir(parents=True)

        print("Training data is being created.")
        train_create_data = functools.partial(
            create_data,
            source_path=TRAIN_DATA_PATH,
            tokenizer=tokenizer,
            is_train=True,
            rolled=True,
            augmented=True,
            num_repeat=args.num_repeat_per_shard,
        )

        training_file_paths = [
            PROCESSED_DATA_PATH / f"training_{i}.npy"
            for i in range(1, args.num_shards + 1)
        ]
        with futures.ProcessPoolExecutor(args.num_workers) as executor:
            for outfile_path in executor.map(train_create_data, training_file_paths):
                print(f"File {outfile_path} is created.")

    # Validation data is being created
    create_data(
        source_path=VAL_DATA_PATH,
        tokenizer=tokenizer,
        output_file_path=PROCESSED_DATA_PATH / "validation_1.npy",
        is_train=False,
        rolled=False,
        augmented=False,
        num_repeat=1,
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
