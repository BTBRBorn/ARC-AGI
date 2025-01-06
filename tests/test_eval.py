import pytest
from eval import Evaluator
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    pass


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
    evaluator = Evaluator(Path('checkpoints/test_model.tar'), Path('data/training'))
    return evaluator

def test_create_context(evaluator, data):
    context_1 = evaluator._create_context(data, 0, evaluator.checkpoint['tokenizer'])
    context_2 = evaluator._create_context(data, 1, evaluator.checkpoint['tokenizer'])

    truth_context_1 = [10, 1, 0, 14, 0, 1, 11, 12, 0, 1, 14, 1, 1, 13, 10, 1, 1, 14, 8, 2, 11, 12, 4, 1, 14, 1, 1, 13,
                       10, 1, 0, 14, 5, 1, 11, 12]

    truth_context_2 = [10, 1, 0, 14, 0, 1, 11, 12, 0, 1, 14, 1, 1, 13, 10, 1, 1, 14, 8, 2, 11, 12, 4, 1, 14, 1, 1, 13,
                       10, 1, 2, 14, 8, 2, 11, 12]

    assert context_1 == truth_context_1
    assert context_2 == truth_context_2