class Tokenizer:
    def __init__(self, vocab_size):
        self.special_tokens = {
            "start_of_input": None,
            "end_of_input": None,
            "start_of_output": None,
            "end_of_output": None,
            "row_indicator": None,
            "context_indicator": None,
        }

        last_token = vocab_size - len(self.special_tokens)
        for key in self.special_tokens.keys():
            self.special_tokens[key] = last_token
            last_token += 1

    def _flatten(self, array, with_rows=False):
        flat_l = []
        if with_rows:
            for i, e in enumerate(array):
                if i != len(array) - 1:
                    flat_l.extend(e + [self.special_tokens["row_indicator"]])
                else:
                    flat_l.extend(e)
        else:
            for e in array:
                flat_l.extend(e)
        return flat_l

    def encode(self, array):
        data = self._flatten(
            [
                [self.special_tokens["start_of_input"]]
                + self._flatten(e["input"], with_rows=True)
                + [self.special_tokens["end_of_input"]]
                + [self.special_tokens["start_of_output"]]
                + self._flatten(e["output"], with_rows=True)
                + [self.special_tokens["end_of_output"]]
                for e in array
            ]
        )

        data = [self.special_tokens['context_indicator']] + data

        return data

    def decode(self, tokens):
        examples = []
        context = None
        for token in tokens:
            if token == self.special_tokens["start_of_input"]:
                example = {"input": []}
                row = []
                context = "input"
            elif token == self.special_tokens["end_of_input"]:
                example["input"].append(row)
            elif token == self.special_tokens["start_of_output"]:
                example["output"] = []
                row = []
                context = "output"
            elif token == self.special_tokens["end_of_output"]:
                example["output"].append(row)
                examples.append(example)
            elif token == self.special_tokens["row_indicator"]:
                example[context].append(row)
                row = []
            elif token == self.special_tokens["context_indicator"]:
                continue
            else:
                row.append(token)

        return examples
