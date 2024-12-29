class Tokenizer:
    def __init__(self, vocab_size):
        self.special_tokens = {
            "start_of_input": None,
            "end_of_input": None,
            "start_of_output": None,
            "end_of_output": None,
            "row_indicator": None,
            "fill_value": None,
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

    def encode(self, array, block_size):
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
        assert (
            len(data) <= block_size
        ), f"Data length ({len(data)}) can't be bigger than block_size ({block_size})"
        # + 1 is needed because buffer size needs to be block_size + 1
        data = data + [self.special_tokens["fill_value"]] * (block_size - len(data) + 1)

        return data
