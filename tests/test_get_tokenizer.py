import pytest
from get_tokenizer import Tokenizer


@pytest.fixture
def data():
    data = {
        "train": [
            {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 1]]},
            {"input": [[1, 1], [8, 2]], "output": [[4, 1], [1, 1]]},
        ],
        "test": [
            {"input": [[1, 0], [5, 1]], "output": [[0, 1], [8, 1]]},
            {"input": [[1, 2], [8, 2]], "output": [[4, 1], [1, 7]]},
        ],
    }

    return data


@pytest.fixture
def tokenizer():
    tokenizer = Tokenizer(64)
    return tokenizer


def test_flatten(tokenizer):
    nested = [list(range(0, 5)), list(range(5, 10)), list(range(10, 15))]
    flattened = list(range(15))
    assert tokenizer._flatten(nested) == flattened


def test_flatten_with_rows(tokenizer):
    array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    new_array = tokenizer._flatten(array, with_rows=True)
    assert new_array == [1, 2, 3, 62, 4, 5, 6, 62, 7, 8, 9]


def test_encode(data, tokenizer):
    data = {
        "train": [
            {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 1]]},
            {"input": [[1, 1], [8, 2]], "output": [[4, 1], [1, 1]]},
        ],
        "test": [
            {"input": [[1, 0], [5, 1]], "output": [[0, 1], [8, 1]]},
            {"input": [[1, 2], [8, 2]], "output": [[4, 1], [1, 7]]},
        ],
    }

    special_tokens = tokenizer.special_tokens

    train_data = (
        [
            special_tokens["start_of_input"],
            1,
            0,
            special_tokens["row_indicator"],
            0,
            1,
            special_tokens["end_of_input"],
            special_tokens["start_of_output"],
            0,
            1,
            special_tokens["row_indicator"],
            1,
            1,
            special_tokens["end_of_output"],
            special_tokens["start_of_input"],
            1,
            1,
            special_tokens["row_indicator"],
            8,
            2,
            special_tokens["end_of_input"],
            special_tokens["start_of_output"],
            4,
            1,
            special_tokens["row_indicator"],
            1,
            1,
            special_tokens["end_of_output"],
            special_tokens["fill_value"],
            special_tokens["fill_value"],
            special_tokens["fill_value"],
        ]
    )

    test_data = (
        [
            special_tokens["start_of_input"],
            1,
            0,
            special_tokens["row_indicator"],
            5,
            1,
            special_tokens["end_of_input"],
            special_tokens["start_of_output"],
            0,
            1,
            special_tokens["row_indicator"],
            8,
            1,
            special_tokens["end_of_output"],
            special_tokens["start_of_input"],
            1,
            2,
            special_tokens["row_indicator"],
            8,
            2,
            special_tokens["end_of_input"],
            special_tokens["start_of_output"],
            4,
            1,
            special_tokens["row_indicator"],
            1,
            7,
            special_tokens["end_of_output"],
            special_tokens["fill_value"],
            special_tokens["fill_value"],
            special_tokens["fill_value"],
        ]
    )

    assert train_data == tokenizer.encode(data["train"], block_size=30)
    assert test_data == tokenizer.encode(data["test"], block_size=30)
    
