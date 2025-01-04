import torch
from copy import deepcopy
from utils import load_checkpoint
import os
from pathlib import Path
import json


class Evaluator:
    def __init__(self, path_to_checkpoint, path_to_tasks):
        self.path_to_checkpoint = Path(path_to_checkpoint)
        self.path_to_tasks = Path(path_to_tasks)
        self.checkpoint = load_checkpoint(path_to_checkpoint)

    def _create_context(self, task, test_index, tokenizer):
        new_task = {}
        new_task["context"] = deepcopy(task["train"])
        test_input = deepcopy(
            {"input": task["test"][test_index]["input"], "output": [[]]}
        )
        new_task["context"].append(test_input)
        tokens = tokenizer.encode(new_task["context"], block_size=None)
        return tokens[:-1]

    def _generate_solution(self, model, task, test_index):
        tokenizer = self.checkpoint["tokenizer"]
        context = self._create_context(task, test_index, tokenizer)
        model.eval()
        pass

    #  with torch.inference_mode():
    #      while tokenizer["end_of_output"] != token:

    def _check_solution(self, output, solution):
        pass

    def _check_pixel_values(self, output, solution):
        pass

    def _finetune(self, task):
        pass

    def evaluate(self, verbose=False):
        model = self.checkpoint['model']
        task_paths = [
            self.path_to_tasks / file for file in os.listdir(self.path_to_tasks)
        ]
        task_acc, pixel_acc = [], []
        for task_path in task_paths:
            with open(task_path, "r") as fhandle:
                task = json.load(fhandle)

            for tx in range(len(task["test"])):
                output = task["test"][tx]["output"]
                solution = self._generate_solution(model, task, tx)
                task_acc.append(self._check_solution(output, solution))
                pixel_acc.append(self._check_pixel_values(output, solution))
                if verbose:
                    print(
                        f"Task {task_path} test {tx+1}: task solved: {task_acc[-1]},"
                        + f"pixel accuracy: {pixel_acc[-1]}"
                    )

        overall_acc = sum(task_acc) / len(task_acc)
        overall_pixel_acc = sum(pixel_acc) / len(pixel_acc)

        print(
            f"Overall accuracy: {overall_acc},"
            + f"Overall pixel accuracy: {overall_pixel_acc}"
        )
