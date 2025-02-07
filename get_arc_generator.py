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


def random_rectangle_zeros(min_size=2, max_size=8):
    n_rows, n_cols = (
        random.randint(min_size, max_size),
        random.randint(min_size, max_size),
    )
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


def impute_diagonal(array, start_position: tuple[int, int], direction):
    if direction == "ne":
        sums = (-1, 1)
    elif direction == "sw":
        sums = (1, -1)
    elif direction == "nw":
        sums = (-1, -1)
    elif direction == "se":
        sums = (1, 1)
    else:
        raise ValueError
    i, j = start_position
    while True:
        try:
            i, j = i + sums[0], j + sums[1]
            if i < 0 or j < 0:
                break
            array[i, j] = 1
        except IndexError:
            break


# ARC task generators starts from this line
def rect_encap(num_repeat, max_rec_size=(8, 8), max_padding=5):
    while True:
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
        yield examples


def rect_encap_v2(num_repeat, max_rec_size=(8, 8), max_padding=5):
    while True:
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
        yield examples


def fill_rectangle(num_repeat):
    while True:
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
        yield examples


def vertical_lines(num_repeat):
    while True:
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
        yield examples


def horizontal_lines(num_repeat):
    while True:
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
        yield examples


def draw_diagonal_1(num_repeat, min_size=4, max_size=20):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size=min_size, max_size=max_size)
            ratio = 0.05
            num_changes = int(ratio * input_array.size)
            num_changes = num_changes if num_changes else 1
            locations = []
            for _ in range(num_changes):
                i, j = random_location(*input_array.shape)
                locations.append((i, j))
                input_array[i, j] = 1
            output_array = np.copy(input_array)
            for location in locations:
                impute_diagonal(output_array, location, "ne")
                impute_diagonal(output_array, location, "sw")
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def draw_diagonal_2(num_repeat, min_size=4, max_size=20):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size=min_size, max_size=max_size)
            ratio = 0.05
            num_changes = int(ratio * input_array.size)
            num_changes = num_changes if num_changes else 1
            locations = []
            for _ in range(num_changes):
                i, j = random_location(*input_array.shape)
                locations.append((i, j))
                input_array[i, j] = 1
            output_array = np.copy(input_array)
            for location in locations:
                impute_diagonal(output_array, location, "nw")
                impute_diagonal(output_array, location, "se")
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def draw_xs(num_repeat, min_size=4, max_size=20):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size=min_size, max_size=max_size)
            ratio = 0.05
            num_changes = int(ratio * input_array.size)
            num_changes = num_changes if num_changes else 1
            locations = []
            for _ in range(num_changes):
                i, j = random_location(*input_array.shape)
                locations.append((i, j))
                input_array[i, j] = 1
            output_array = np.copy(input_array)
            for location in locations:
                impute_diagonal(output_array, location, "ne")
                impute_diagonal(output_array, location, "sw")
                impute_diagonal(output_array, location, "nw")
                impute_diagonal(output_array, location, "se")
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def color_mapping_columns(num_repeat, min_size=3, max_size=8):
    colors = list(range(1, 10))
    random.shuffle(colors)
    mappings = {colors.pop(): colors.pop() for i in range(4)}
    for k, v in tuple(mappings.items()):
        mappings[v] = k
    mappings[colors[-1]] = colors[-1]
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size, max_size)
            output_array = np.copy(input_array)
            n_cols = input_array.shape[1]
            for j in range(n_cols):
                input_color = random.randint(1, 9)
                input_array[:, j] = input_color
                output_array[:, j] = mappings[input_color]
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def color_mapping_rows(num_repeat, min_size=3, max_size=8):
    colors = list(range(1, 10))
    random.shuffle(colors)
    mappings = {colors.pop(): colors.pop() for i in range(4)}
    for k, v in tuple(mappings.items()):
        mappings[v] = k
    mappings[colors[-1]] = colors[-1]
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size, max_size)
            output_array = np.copy(input_array)
            n_rows = input_array.shape[0]
            for i in range(n_rows):
                input_color = random.randint(1, 9)
                input_array[i, :] = input_color
                output_array[i, :] = mappings[input_color]
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def color_mapping_dots(num_repeat, min_size=4, max_size=10):
    colors = list(range(1, 10))
    random.shuffle(colors)
    mappings = {colors.pop(): colors.pop() for i in range(4)}
    for k, v in tuple(mappings.items()):
        mappings[v] = k
    mappings[colors[-1]] = colors[-1]
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size, max_size)
            output_array = np.copy(input_array)
            n_dots = int(0.2 * input_array.size)
            n_dots = n_dots if n_dots else 1
            for i in range(n_dots):
                input_color = random.randint(1, 9)
                i, j = random_location(*input_array.shape)
                input_array[i, j] = input_color
                output_array[i, j] = mappings[input_color]
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


class ArcGenerator:
    def __init__(self, num_repeat: int | Iterable[int]):
        self.task_generators = [
            rect_encap(num_repeat),
            rect_encap_v2(num_repeat),
            fill_rectangle(num_repeat),
            vertical_lines(num_repeat),
            horizontal_lines(num_repeat),
            draw_diagonal_1(num_repeat),
            draw_diagonal_2(num_repeat),
            draw_xs(num_repeat),
            color_mapping_columns(num_repeat=4),
            color_mapping_rows(num_repeat=4),
            color_mapping_dots(num_repeat=4),
        ]

    def __call__(self):
        tasks = []
        for task_generator in self.task_generators:
            tasks.append({"train": next(task_generator)})

        random.shuffle(tasks)
        return tasks
