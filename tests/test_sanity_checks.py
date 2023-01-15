import pytest
import torch

import torch_benchmarks as tbm
from tests.fixtures.models import SimpleModel


def test_cuda() -> None:
    with pytest.raises(EnvironmentError):
        _ = tbm.benchmark(SimpleModel, torch.ones(2), loss=torch.nn.BCELoss())
