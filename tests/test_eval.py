import pytest
from eval import Evaluator
from pathlib import Path


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
def evaluator():
    evaluator = Evaluator(Path("test_checkpoints/model.tar"), Path("data/training"))
    return evaluator


def test_create_context(evaluator, data):
    context_1 = evaluator._create_context(data, 0, evaluator.checkpoint["tokenizer"])
    context_2 = evaluator._create_context(data, 1, evaluator.checkpoint["tokenizer"])

    # tokenizer with vocab_size=16 is assumed with truth_contexts
    truth_context_1 = [
        15,
        10,
        1,
        0,
        14,
        0,
        1,
        11,
        12,
        0,
        1,
        14,
        1,
        1,
        13,
        10,
        1,
        1,
        14,
        8,
        2,
        11,
        12,
        4,
        1,
        14,
        1,
        1,
        13,
        10,
        1,
        0,
        14,
        5,
        1,
        11,
        12,
    ]

    truth_context_2 = [
        15,
        10,
        1,
        0,
        14,
        0,
        1,
        11,
        12,
        0,
        1,
        14,
        1,
        1,
        13,
        10,
        1,
        1,
        14,
        8,
        2,
        11,
        12,
        4,
        1,
        14,
        1,
        1,
        13,
        10,
        1,
        2,
        14,
        8,
        2,
        11,
        12,
    ]

    assert context_1 == truth_context_1
    assert context_2 == truth_context_2


def test_is_2d_array(evaluator):
    assert evaluator._is_2d_array([[1], [2]]) == bool(1)
    assert evaluator._is_2d_array([[3]]) == bool(1)
    assert evaluator._is_2d_array([[2, 3], [3]]) == bool(0)
    assert evaluator._is_2d_array([[1], [1, 2, 3], [2, 3]]) == bool(0)


def test_check_pixel_values(evaluator):
    assert evaluator._check_pixel_values([[3]], None) == 0.0
    assert evaluator._check_pixel_values([[1, 2], [4, 4]], [[1, 2], [4, 4]]) == 1.0
    assert evaluator._check_pixel_values([[1, 2], [4, 4]], [[1, 2], [4, 5]]) == 0.75
    assert evaluator._check_pixel_values([[1, 2], [4, 4]], [[7, 2], [4, 5]]) == 0.50
    assert evaluator._check_pixel_values([[1, 2, 3], [4, 4, 6]], [[7, 2], [4, 5]]) == 0.0