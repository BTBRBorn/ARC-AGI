import torch
import torch.nn.functional as F
from copy import deepcopy
from utils import load_checkpoint
import os
from pathlib import Path
import json
import argparse
from collections import deque
import math


class Evaluator:
    def __init__(self, path_to_checkpoint, path_to_tasks, k_beam, device):
        self.path_to_checkpoint = Path(path_to_checkpoint)
        self.path_to_tasks = Path(path_to_tasks)
        self.k_beam = k_beam
        self.device = device
        self.checkpoint = load_checkpoint(
            path_to_checkpoint,
            device,
            compile_model=False,
            with_model=True,
            ddp_model=False,
        )

    def _create_context(self, task, test_index, tokenizer):
        new_task = {}
        new_task["context"] = deepcopy(task["train"])
        test_input = deepcopy(
            {"input": task["test"][test_index]["input"], "output": [[]]}
        )
        new_task["context"].append(test_input)
        tokens = tokenizer.encode(new_task["context"])
        return tokens[:-1], len(tokens[:-1])

    @staticmethod
    def _bfs(
        model,
        context: torch.Tensor,
        config,
        tokenizer,
        tokens_threshold,
        prob_threshold=1.0,
    ):
        contexts = deque([(context, 0.0, 0)])
        solutions = []
        with torch.inference_mode():
            while contexts:
                context, score, counter = contexts.popleft()
                if counter > tokens_threshold:
                    continue
                context = context[:, -config.block_size :]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(context)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                # Bundle next_tokens and their probability together
                probs = probs.view(-1)
                mask = probs > prob_threshold
                tokens = mask.nonzero(as_tuple=True)[0]
                if len(tokens) == 0:
                    tokens = torch.argmax(probs).unsqueeze(dim=0)
                    mask = tokens
                next_tokens = zip(tokens, probs[mask])
                for next_token, prob in next_tokens:
                    next_context = torch.cat((context, next_token.view(1, -1)), dim=-1)
                    new_score = score + math.log(prob.item())
                    new_counter = counter + 1
                    if next_token.item() != tokenizer.special_tokens["end_of_output"]:
                        contexts.append((next_context, new_score, new_counter))
                    else:
                        solution = tokenizer.decode(
                            next_context.tolist()[0], only_last_output=True
                        )[0]["output"]
                        new_score /= len(solution)
                        solutions.append((solution, new_score))
        return solutions

    @staticmethod
    def _beam_search(
        model: torch.nn.Module,
        context: torch.Tensor,
        config,
        tokenizer,
        tokens_threshold,
        k_beam,
    ):
        # canditates = [(context, score, is_finished)]
        beams = [(context, 0.0, False)]
        end_of_output = tokenizer.special_tokens["end_of_output"]
        with torch.inference_mode():
            for _ in range(tokens_threshold):
                candidates = []
                not_finished_seqs = [seq for seq, _, finished in beams if not finished]
                # If every beam ends with end_of_output token break early
                if len(not_finished_seqs) == 0:
                    break
                context = torch.cat(not_finished_seqs, dim=0)
                context = context[:, -config.block_size :]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(context)[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                top_log_probs, top_tokens = torch.topk(log_probs, k=k_beam, dim=-1)

                i = 0
                for seq, score, _ in beams:
                    if seq[0, -1].item() == end_of_output:
                        candidates.append((seq, score, True))
                        continue

                    for log_prob, token in zip(top_log_probs[i], top_tokens[i]):
                        next_seq = torch.cat((seq, token.view(1, 1)), dim=-1)
                        next_score = score + log_prob.item()
                        candidates.append(
                            (
                                next_seq,
                                next_score,
                                True if token.item() == end_of_output else False,
                            )
                        )
                    i += 1

                beams = sorted(
                    candidates,
                    key=lambda x: x[1] / x[0].shape[1],
                    reverse=True,
                )[:k_beam]

        solutions = [
            (
                tokenizer.decode(seq.tolist()[0], only_last_output=True)[0]["output"],
                score / seq.shape[1],
            )
            for seq, score, finished in beams
            if finished
        ]

        return solutions

    def _generate_solutions(
        self,
        model,
        task,
        test_index,
        tokens_threshold=2000,
    ):
        tokenizer = self.checkpoint["tokenizer"]
        config = self.checkpoint["config"]
        context, con_len = self._create_context(task, test_index, tokenizer)
        context = torch.tensor(context, device=self.device).view(1, -1)
        solutions = self._beam_search(
            model,
            context,
            config,
            tokenizer,
            tokens_threshold,
            self.k_beam,
        )

        if len(solutions) == 0:
            return None, con_len
        else:
            return [
                s[0] for s in sorted(solutions, key=lambda x: x[1], reverse=True)[:2]
            ], con_len

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
        for task_number, task_path in enumerate(task_paths, start=1):
            with open(task_path, "r") as fhandle:
                task = json.load(fhandle)

            for tx in range(len(task["test"])):
                output = task["test"][tx]["output"]

                solutions, con_len = self._generate_solutions(model, task, tx)
                acc = (
                    max(self._check_solution(output, s) for s in solutions)
                    if solutions is not None
                    else 0.0
                )
                p_acc = (
                    max(self._check_pixel_values(output, s) for s in solutions)
                    if solutions is not None
                    else 0.0
                )
                task_acc.append(acc)
                pixel_acc.append(p_acc)
                if verbose:
                    print(
                        f"Task {task_number!s:>3s}/{total_tasks} test {tx + 1}, "
                        + f"context length: {con_len!s:>4s}, task solved: {task_acc[-1]}, "
                        + f"pixel accuracy percentage: {pixel_acc[-1] * 100:.2f}%"
                    )

        overall_acc = (sum(task_acc) / len(task_acc)) * 100
        overall_pixel_acc = (sum(pixel_acc) / len(pixel_acc)) * 100

        print(
            f"Overall accuracy: {overall_acc:.2f}%, "
            + f"Overall pixel accuracy: {overall_pixel_acc:.2f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--tasks_path", type=str)
    parser.add_argument("--verbose", type=int, choices={0, 1}, default=1)
    parser.add_argument("--k_beam", type=int, default=1)

    args = parser.parse_args()

    device = torch.device("cuda:0")
    evaluator = Evaluator(
        args.checkpoint_path, args.tasks_path, args.k_beam, device=device
    )
    evaluator.evaluate(verbose=bool(args.verbose))
