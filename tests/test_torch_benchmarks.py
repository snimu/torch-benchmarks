import pytest
import torch

from tests.fixtures.models import SimpleModel
from torch_benchmarks import benchmark


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="`benchmark` only works on CUDA."
)
def test_doesnt_measure_cuda_itself() -> None:
    result = benchmark(SimpleModel, input_data=torch.ones(2), loss=torch.nn.BCELoss())
    assert result.memory_bytes_forward_backward < 1e6
