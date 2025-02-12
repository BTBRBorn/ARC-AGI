import torch
from copy import deepcopy
from utils import load_checkpoint
import os
from pathlib import Path
import json
import argparse


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
        tokens = tokenizer.encode(new_task["context"])
        return tokens[:-1], len(tokens[:-1])

    def _generate_solution(self, model, task, test_index, threshold=2000):
        tokenizer = self.checkpoint["tokenizer"]
        config = self.checkpoint["config"]
        context, con_len = self._create_context(task, test_index, tokenizer)
        context = torch.tensor(context, device=config.device).view(1, -1)
        model.eval()
        counter = 0
        with torch.inference_mode():
            with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                next_token = -1
                while (
                    tokenizer.special_tokens["end_of_output"] != next_token
                    and counter <= threshold
                ):
                    counter += 1
                    context = context[:, -config.block_size :]
                    logits = model(context)
                    next_token = torch.argmax(logits[:, -1, :])
                    context = torch.cat((context, next_token.view(1, -1)), dim=-1)
                    next_token = next_token.item()
                tokens = context.view(-1).tolist()
        if counter > threshold:
            return None, con_len

        tokens = tokenizer.decode(tokens, only_last_output=True)
        
        return tokens[0]["output"], con_len

    def _check_solution(self, output, solution):
        if solution is None:
            return False
        return output == solution

    def _is_2d_array(self, array):
        lengths = set([len(e) for e in array])
        return len(lengths) == 1

    def _check_pixel_values(self, output, solution):
        if solution is None:
            return 0.0
        if self._is_2d_array(solution) and (
            len(output) == len(solution) and len(output[0]) == len(solution[0])
        ):
            total_pixels = len(output) * len(output[0])
            matched_pixels = 0
            for i in range(len(output)):
                for j in range(len(output[0])):
                    if output[i][j] == solution[i][j]:
                        matched_pixels += 1
            return matched_pixels / total_pixels
        else:
            return 0.0

    def evaluate(self, verbose=False):
        model = self.checkpoint["model"]
        task_paths = [
            self.path_to_tasks / file for file in os.listdir(self.path_to_tasks)
        ]
        task_acc, pixel_acc = [], []
        total_tasks = len(task_paths)
        for task_number, task_path in enumerate(task_paths):
            with open(task_path, "r") as fhandle:
                task = json.load(fhandle)

            for tx in range(len(task["test"])):
                output = task["test"][tx]["output"]
                ###finetune should be done here###
                solution, con_len = self._generate_solution(model, task, tx)
                task_acc.append(self._check_solution(output, solution))
                pixel_acc.append(self._check_pixel_values(output, solution))
                if verbose:
                    print(
                        f"Task {task_number!s:>3s}/{total_tasks} test {tx + 1}, " 
                        + f"context length: {con_len!s:>4s}, task solved: {task_acc[-1]}, "
                        + f"pixel accuracy percentage: {pixel_acc[-1] * 100:.2f}%"
                    )

        overall_acc = (sum(task_acc) / len(task_acc)) * 100
        overall_pixel_acc = sum(pixel_acc) / len(pixel_acc) * 100

        print(
            f"Overall accuracy: {overall_acc:.2f}%, "
            + f"Overall pixel accuracy: {overall_pixel_acc:.2f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--tasks_path", type=str)
    parser.add_argument("--verbose", type=int, choices={0, 1}, default=1)

    args = parser.parse_args()

    evaluator = Evaluator(args.checkpoint_path, args.tasks_path)
    evaluator.evaluate(verbose=bool(args.verbose))
