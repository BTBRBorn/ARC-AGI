import pytest
from data import prepare
import numpy as np


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


def test_flatten():
    nested = [list(range(0, 5)), list(range(5, 10)), list(range(10, 15))]
    flattened = list(range(15))
    assert prepare.flatten(nested) == flattened


def test_create_array_without_special(data):
    train_data_without_special = np.array(
        [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 8, 2, 4, 1, 1, 1], dtype=np.uint16
    )
    test_data_without_special = np.array(
        [1, 0, 5, 1, 0, 1, 8, 1, 1, 2, 8, 2, 4, 1, 1, 7], dtype=np.uint16
    )

    assert (train_data_without_special == prepare.create_array(data)).all()
    assert (
        test_data_without_special == prepare.create_array(data, is_train=False)
    ).all()


def test_create_data_with_special_tokens(data):
    special_tokens = {
        "start_of_input": 24,
        "end_of_input": 25,
        "start_of_output": 26,
        "end_of_output": 27,
        "start_of_example": 28,
        "end_of_example": 29,
        "start_of_task": 30,
        "end_of_task": 31,
    }

    train_data = np.array(
        [
            30,
            28,
            24,
            1,0,0,1,
            25,
            26,
            0,1,1,1,
            27,
            29,
            28,
            24,
            1,1,8,2,
            25,
            26,
            4,1,1,1,
            27,
            29,
            31,
        ],
        dtype=np.uint16,
    )

    test_data = np.array(
        [
            30,
            28,
            24,
            1,0,5,1,
            25,
            26,
            0,1,8,1,
            27,
            29,
            28,
            24,
            1,2,8,2,
            25,
            26,
            4,1,1,7,
            27,
            29,
            31,
        ],
        dtype=np.uint16,
    )

    assert (
        train_data
        == prepare.create_array(data, special_tokens, add_special_tokens=True)
    ).all()
    assert (
        test_data
        == prepare.create_array(
            data, special_tokens, add_special_tokens=True, is_train=False
        )
    ).all()
