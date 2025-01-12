import pytest
from unittest import mock
from get_augmentor import Augmentor
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
def augmentor():
    vocab_size = 256
    tokenizer = Tokenizer(vocab_size)
    augmentor = Augmentor(vocab_size, tokenizer.special_tokens)
    return augmentor


def test_flatten(augmentor):
    nested = [list(range(0, 5)), list(range(5, 10)), list(range(10, 15))]
    flattened = list(range(15))
    assert augmentor._flatten(nested) == flattened


def test_get_mappings(augmentor, data):
    with mock.patch("random.choice", side_effect= [10, 11, 12, 13, 14, 15]):
        mappings_1 = augmentor._get_mappings(data["train"][0])
        mappings_2 = augmentor._get_mappings(data["train"][1])
    assert mappings_1 == {0: 10, 1: 11}
    assert mappings_2 == {1: 13, 8: 12, 2: 14, 4: 15}


def test_change_array(augmentor, data):
    mappings = {1: 11, 2: 12, 8: 14}
    # array = deepcopy(data['train'][1]['input'])
    array = data["train"][1]["input"]
    augmentor._change_array(array, mappings)
    assert array == [[11, 11], [14, 12]]


def test_change_one_example(augmentor, data):
    example = data["test"][1]
    with mock.patch(
        "get_augmentor.Augmentor._get_mappings",
        return_value={1: 11, 2: 12, 4: 13, 7: 13, 8: 15},
    ):
        augmentor._change_one_example(example)
    assert example == {"input": [[11, 12], [15, 12]], "output": [[13, 11], [11, 13]]}


def test_change_colors(augmentor, data):
    task = data["train"]
    with mock.patch(
        "get_augmentor.Augmentor._get_mappings",
        return_value={1: 11, 2: 12, 4: 13, 7: 13, 8: 15, 0:16},
    ):
        augmentor._change_colors(task)
    assert task == [
        {"input": [[11, 16], [16, 11]], "output": [[16, 11], [11, 11]]},
        {"input": [[11, 11], [15, 12]], "output": [[13, 11], [11, 11]]},
    ]
