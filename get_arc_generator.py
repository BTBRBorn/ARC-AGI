from collections.abc import Iterable
import numpy as np
import random

def trim_array(array, *, left=0, right=0, up=0, down=0):
    down = None if down == 0 else -down
    right = None if right == 0 else -right
    up = None if up == 0 else up
    left = None if left == 0 else left
    return array[up:down, left:right]

def rect_encap(num_repeat, max_rec_size=(8,8), max_padding=5):
    examples = []
    for _ in range(num_repeat):
        n_rows, n_cols = random.randint(1, max_rec_size[0]), random.randint(1, max_rec_size[1])
        output_array = np.ones(shape=(n_rows, n_cols))
        num_pad = random.randint(1, max_padding)
        input_array = np.pad(output_array, pad_width=num_pad)
        x_shift, y_shift =random.randint(-num_pad, num_pad), random.randint(-num_pad, num_pad) 
        input_array = np.roll(input_array, shift=(x_shift, y_shift), axis=(0,1))
        examples.append({'input':input_array.tolist(), 'output':output_array.tolist()})
    return examples

def rect_encap_v2(num_repeat, max_rec_size=(8,8), max_padding=5):
    examples = []
    for _ in range(num_repeat):
        n_rows, n_cols = random.randint(1, max_rec_size[0]), random.randint(1, max_rec_size[1])
        #This one has random output array
        output_array = np.random.randint(low=1, high=10, size=(n_rows, n_cols))
        num_pad = random.randint(1, max_padding)
        input_array = np.pad(output_array, pad_width=num_pad)
        x_shift, y_shift =random.randint(-num_pad, num_pad), random.randint(-num_pad, num_pad) 
        input_array = np.roll(input_array, shift=(x_shift, y_shift), axis=(0,1))
        examples.append({'input':input_array.tolist(), 'output':output_array.tolist()})
    return examples

class ArcGenerator:

    def __init__(self):
        self.task_generators = [rect_encap, rect_encap_v2]

    def __call__(self, num_repeats: int | Iterable[int]):

        if not isinstance(num_repeats, Iterable):
            num_repeats = (num_repeats, ) * len(self.task_generators)

        tasks = []
        for num_repeat, task_generator in zip(num_repeats, self.task_generators):
            tasks.append({'train':task_generator(num_repeat)})

        return tasks
