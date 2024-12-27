import json
from pathlib import Path
import numpy as np
import argparse
from dataclasses import dataclass
import os


def flatten(nested: list[list]) -> list:
    flat_l = []
    for e in nested:
        flat_l.extend(e)
    return flat_l


def create_array(
    json_file, special_tokens=None, is_train=True
):
    if is_train:
        json_file = json_file["train"]
    else:
        json_file = json_file["test"]

    if special_tokens is not None:
        data = flatten(
            [
                [special_tokens["start_of_input"]]
                + flatten(e["input"])
                + [special_tokens["end_of_input"]]
                + [special_tokens["start_of_output"]]
                + flatten(e["output"])
                + [special_tokens["end_of_output"]]
                for e in json_file
            ]
        )

        data = (
            [special_tokens["start_of_task"]] + data + [special_tokens["end_of_task"]]
        )

    else:
        data = flatten([flatten(e["input"]) + flatten(e["output"]) for e in json_file])

    return np.array(data, dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/training/")

    args = parser.parse_args()

    @dataclass
    class Config:
        data_path: str = args.data_path

    config = Config()

    data_path = Path(config.data_path)
    file_list = os.listdir(data_path)

    array_list = []
    for file in file_list:
        with open(data_path / file, "r") as fhandle:
            json_file = json.load(fhandle)
        array_list.append(create_array(json_file))
    all_values = np.concatenate(array_list)

    print(f'Unique values inside training data: {set(int(d) for d in all_values)}')

    max_value = int(all_values.max())

    special_tokens = {
        "start_of_input": None,
        "end_of_input": None,
        "start_of_output": None,
        "end_of_output": None,
        "start_of_task": None,
        "end_of_task": None,
    }

    for key in special_tokens.keys():
        special_tokens[key] = max_value + 1
        max_value += 1

    def create_dataset(save_path, file_list, name, is_train=True):
        array_list = []
        for file in file_list:
            with open(data_path / file, "r") as fhandle:
                json_file = json.load(fhandle)
            array_list.append(
                create_array(
                    json_file, special_tokens=special_tokens, is_train=is_train
                )
            )
        all_tokens = np.concatenate(array_list)
        np.save(Path(save_path) / name, all_tokens)
    
    pretraining_path = Path("data/pretraining")

    pretraining_path.mkdir(exist_ok=True, parents=True)

    create_dataset(pretraining_path, file_list, "training.npy")
    create_dataset(pretraining_path, file_list, "validation.npy", is_train=False)

    with open(pretraining_path / 'special_tokens.json', 'w') as fhandle:
        json.dump(special_tokens, fhandle)
