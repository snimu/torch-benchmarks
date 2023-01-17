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

    @staticmethod
    def test_combined_size_is_full_size_forward_only() -> None:
        device = torch.device("cuda:0")
        model_type = torch.nn.Linear
        model_args = (10, 10)
        loss = torch.nn.CrossEntropyLoss()

        # measure full size
        bytes_before = torch.cuda.max_memory_allocated(device)

        with torch.no_grad():
            input_data = torch.ones(10).to(device)
            model = model_type(*model_args).to(device)
            model(input_data)

        bytes_full_measurement = torch.cuda.max_memory_allocated(device) - bytes_before
        del model, input_data

        # measure full size from two independent components for comparison
        model_statistics = benchmark(
            model_type, torch.ones(10), loss, model_args=model_args
        )

        bytes_before = torch.cuda.max_memory_allocated(device)
        input_data = torch.ones(10).to(device)
        bytes_input_data = torch.cuda.max_memory_allocated(device) - bytes_before
        del input_data

        # do comparison
        bytes_full_benchmark = bytes_input_data + model_statistics.memory_bytes_forward

        assert (
            abs(bytes_full_benchmark - bytes_full_measurement) / bytes_full_measurement
            < 0.01
        )
