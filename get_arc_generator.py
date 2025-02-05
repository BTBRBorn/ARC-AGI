from collections.abc import Iterable
import numpy as np
import random


# Helper Functions
def trim_array(array, *, left=0, right=0, up=0, down=0):
    down = None if down == 0 else -down
    right = None if right == 0 else -right
    up = None if up == 0 else up
    left = None if left == 0 else left
    return array[up:down, left:right]


def random_rectangle_zeros(max_size=8):
    n_rows, n_cols = random.randint(2, max_size), random.randint(2, max_size)
    return np.zeros(shape=(n_rows, n_cols))


def random_pad(array, max_padding):
    num_pad = random.randint(1, max_padding)
    out_array = np.pad(array, pad_width=num_pad)
    return out_array, num_pad


def random_roll(array, max_shift):
    x_shift, y_shift = (
        random.randint(-max_shift, max_shift),
        random.randint(-max_shift, max_shift),
    )
    array = np.roll(array, shift=(x_shift, y_shift), axis=(0, 1))
    return array, (x_shift, y_shift)


def random_location(n_rows, n_cols):
    return random.randint(0, n_rows - 1), random.randint(0, n_cols - 1)


# ARC task generators starts from this line
def rect_encap(num_repeat, max_rec_size=(8, 8), max_padding=5):
    examples = []
    for _ in range(num_repeat):
        n_rows, n_cols = (
            random.randint(1, max_rec_size[0]),
            random.randint(1, max_rec_size[1]),
        )
        output_array = np.ones(shape=(n_rows, n_cols))
        # pad the output_array to get input array
        input_array, num_pad = random_pad(output_array, max_padding)
        # roll the output array
        input_array, shifts = random_roll(input_array, num_pad)
        examples.append(
            {"input": input_array.tolist(), "output": output_array.tolist()}
        )
    return examples


def rect_encap_v2(num_repeat, max_rec_size=(8, 8), max_padding=5):
    examples = []
    for _ in range(num_repeat):
        n_rows, n_cols = (
            random.randint(1, max_rec_size[0]),
            random.randint(1, max_rec_size[1]),
        )
        # This one has random output array
        output_array = np.random.randint(low=1, high=10, size=(n_rows, n_cols))
        # pad the output_array to get input array
        input_array, num_pad = random_pad(output_array, max_padding)
        # roll the output array
        input_array, shifts = random_roll(input_array, num_pad)
        examples.append(
            {"input": input_array.tolist(), "output": output_array.tolist()}
        )
    return examples


def fill_rectangle(num_repeat):
    examples = []
    for _ in range(num_repeat):
        zeros = random_rectangle_zeros()
        input_array = np.pad(zeros, pad_width=1, constant_values=1)
        output_array = np.where(input_array == 0, 2, input_array)
        input_array, num_pad = random_pad(input_array, max_padding=5)
        input_array, shifts = random_roll(input_array, max_shift=num_pad)
        output_array = np.pad(output_array, pad_width=num_pad)
        output_array = np.roll(output_array, shift=shifts, axis=(0, 1))
        examples.append(
            {"input": input_array.tolist(), "output": output_array.tolist()}
        )
    return examples


def vertical_lines(num_repeat):
    examples = []
    for _ in range(num_repeat):
        input_array = random_rectangle_zeros(max_size=10)
        n_rows, n_cols = input_array.shape
        num_lines = random.randint(1, n_cols - 1)
        js = []
        for _ in range(num_lines):
            i, j = random_location(n_rows, n_cols)
            js.append(j)
            input_array[i, j] = 1
        output_array = np.copy(input_array)
        for j in js:
            output_array[:, j] = 1
        examples.append(
            {"input": input_array.tolist(), "output": output_array.tolist()}
        )
    return examples


def horizontal_lines(num_repeat):
    examples = []
    for _ in range(num_repeat):
        input_array = random_rectangle_zeros(max_size=10)
        n_rows, n_cols = input_array.shape
        num_lines = random.randint(1, n_rows - 1)
        rows = []
        for _ in range(num_lines):
            i, j = random_location(n_rows, n_cols)
            rows.append(i)
            input_array[i, j] = 1
        output_array = np.copy(input_array)
        for i in rows:
            output_array[i, :] = 1
        examples.append(
            {"input": input_array.tolist(), "output": output_array.tolist()}
        )
    return examples


class ArcGenerator:
    def __init__(self):
        self.task_generators = [
            rect_encap,
            rect_encap_v2,
            fill_rectangle,
            vertical_lines,
            horizontal_lines,
        ]

    def __call__(self, num_repeats: int | Iterable[int]):
        if not isinstance(num_repeats, Iterable):
            num_repeats = (num_repeats,) * len(self.task_generators)

        tasks = []
        for num_repeat, task_generator in zip(num_repeats, self.task_generators):
            tasks.append({"train": task_generator(num_repeat)})

        return tasks
