# pylint: disable=too-few-public-methods
import pytest
import torch

from tests.fixtures.models import SimpleModel
from torch_benchmarks import benchmark


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="`benchmark` only works on CUDA."
)
class TestTorchBenchmarks:
    """
    Tests for torch-benchmarks.
    In class so that `@pytest.mark.skipif(...) only has to be applied once.
    """

    @staticmethod
    def test_doesnt_measure_cuda_itself() -> None:
        result = benchmark(
            SimpleModel, input_data=torch.ones(2), loss=torch.nn.CrossEntropyLoss()
        )
        assert result.memory_bytes_forward_backward < 1e6
