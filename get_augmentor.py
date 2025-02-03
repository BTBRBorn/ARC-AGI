import random


class Augmentor:
    def __init__(self):
        self.colors = list(range(0, 10))

    def _flatten(self, array):
        flat_l = []
        for e in array:
            flat_l.extend(e)
        return flat_l

    def _get_mappings(self, task):
        flattened = []
        for example in task:
            flattened.extend(
                self._flatten(example["input"]) + self._flatten(example["output"])
            )
        color_set = set(flattened)
        mappings = {}
        copy_colors = list(self.colors)
        for c in color_set:
            new_c = random.choice(copy_colors)
            mappings[c] = new_c
            copy_colors.remove(new_c)
        return mappings

    def _change_array(self, array, mappings):
        n_rows, n_columns = len(array), len(array[0])
        for i in range(n_rows):
            for j in range(n_columns):
                array[i][j] = mappings[array[i][j]]

    def _change_one_example(self, example, mappings):
        self._change_array(example["input"], mappings)
        self._change_array(example["output"], mappings)

    def _change_colors(self, task: list[dict]):
        mappings = self._get_mappings(task)
        for example in task:
            self._change_one_example(example, mappings)

    def _add_noise_array(self, array, ratio):
        n_rows, n_cols = len(array), len(array[0])
        num_changes = int(n_rows * n_cols * ratio)
        for _ in range(num_changes):
            i, j = random.randint(0, n_rows-1), random.randint(0, n_cols-1)
            random_value = random.randint(0, 9)
            array[i][j] = random_value

    def _add_noise_example(self, example, ratio):
        self._add_noise_array(example['input'], ratio)
        self._add_noise_array(example['output'], ratio)
    
    def _add_noise_task(self, task, ratio=0.5):
        for example in task:
            self._add_noise_example(example, ratio)

    def __call__(self, task, change_colors=True, add_noise=False):
        if change_colors:
            self._change_colors(task)
        if add_noise:
            self._add_noise_task(task)
