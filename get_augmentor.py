import random

class Augmentor:
    def __init__(self, vocab_size, special_tokens):

        self.special_tokens = special_tokens
        self.colors = list(
            set(range(0, vocab_size)) - set(self.special_tokens.values())
        )

    def _flatten(self, array):
        flat_l = []
        for e in array:
            flat_l.extend(e)
        return flat_l

    def _get_mappings(self, example):
        flattened = self._flatten(example["input"]) + self._flatten(example["output"])
        color_set = set(flattened)#.difference({0})
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
                #if array[i][j] != 0:
                array[i][j] = mappings[array[i][j]]

    def _change_one_example(self, example):
        mappings = self._get_mappings(example)
        self._change_array(example["input"], mappings)
        self._change_array(example["output"], mappings)

    def _change_colors(self, task: list[list]):
        for example in task:
            self._change_one_example(example)

    def apply(self, task):
            self._change_colors(task)