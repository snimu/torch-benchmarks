import pytest
import torch

from tests.fixtures.inputs import InputsNestedTo
from torch_benchmarks.benchmark import nested_to


class TestNestedTo:
    """Test the `nested_to`-function used for `benchmark`."""

    inputs = InputsNestedTo()

    def test_nested_to_dtype(self) -> None:
        tensor = nested_to(self.inputs.tensor, torch.int8)
        assert tensor.dtype == torch.int8

        self.inputs.nested_tensor = nested_to(self.inputs.nested_tensor, torch.int8)
        for tensor in self.inputs.retrieve_tensors_from_nested_tensor():
            assert tensor.dtype == torch.int8

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Tests moving to CUDA.")
    def test_nested_to_device(self) -> None:
        tensor = nested_to(self.inputs.tensor, "cuda")
        assert tensor.is_cuda

        self.inputs.nested_tensor = nested_to(self.inputs.nested_tensor, "cuda")
        for tensor in self.inputs.retrieve_tensors_from_nested_tensor():
            assert tensor.is_cuda
