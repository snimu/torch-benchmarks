import pytest
import torch
import torchvision as tv  # type: ignore[import]

from tests.fixtures.models import SimpleModel
from torch_benchmarks import benchmark


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="Tests only meaningful without CUDA."
)
def test_sanity_check_cuda() -> None:
    with pytest.raises(EnvironmentError):
        _ = benchmark(SimpleModel, torch.ones(2), loss=torch.nn.CrossEntropyLoss())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="`benchmark` only works on CUDA."
)
class TestSanityChecks:
    """
    Tests for sanity-checks of torch-benchmarks::benchmark.
    In class so that `@pytest.mark.skipif(...) only has to be applied once.
    """

    resnet18 = tv.models.resnet18
    model_kwargs18 = {"weights": tv.models.ResNet18_Weights.IMAGENET1K_V1}
    input_data = torch.randn(4, 3, 200, 200)
    loss = torch.nn.CrossEntropyLoss()

    def test_sanity_check_device(self) -> None:
        valid_devices = [0, "cuda", torch.device("cuda:0")]

        for device in valid_devices:
            benchmark(
                self.resnet18,
                self.input_data,
                self.loss,
                model_kwargs=self.model_kwargs18,
                device=device,  # type: ignore[arg-type]
            )

        # Test default arg
        benchmark(
            self.resnet18, self.input_data, self.loss, model_kwargs=self.model_kwargs18
        )

        with pytest.raises(TypeError):
            benchmark(
                self.resnet18,
                self.input_data,
                self.loss,
                model_kwargs=self.model_kwargs18,
                device=(1, 1),  # type: ignore[arg-type]
            )

        with pytest.raises(ValueError):
            benchmark(
                self.resnet18,
                self.input_data,
                self.loss,
                model_kwargs=self.model_kwargs18,
                device="cpu",
            )

        with pytest.raises(ValueError):
            benchmark(
                self.resnet18,
                self.input_data,
                self.loss,
                model_kwargs=self.model_kwargs18,
                device=torch.device("cpu"),
            )

        with pytest.raises(ValueError):
            benchmark(
                self.resnet18,
                self.input_data,
                self.loss,
                model_kwargs=self.model_kwargs18,
                device=torch.cuda.device_count(),
            )

    def test_sanity_check_num_samples(self) -> None:
        wrong_inputs = [None, 2.0, "not an int"]

        for wrong_input in wrong_inputs:
            with pytest.raises(TypeError):
                benchmark(
                    self.resnet18,
                    self.input_data,
                    self.loss,
                    model_kwargs=self.model_kwargs18,
                    num_samples=wrong_input,  # type: ignore[arg-type]
                )

        with pytest.raises(ValueError):
            benchmark(
                self.resnet18,
                self.input_data,
                self.loss,
                model_kwargs=self.model_kwargs18,
                num_samples=-1,
            )

    def test_sanity_check_model_args(self) -> None:
        benchmark(
            self.resnet18,
            self.input_data,
            self.loss,
            model_args=[],
            model_kwargs=self.model_kwargs18,
        )

        with pytest.raises(TypeError):
            benchmark(
                self.resnet18,
                self.input_data,
                self.loss,
                model_args={"foo": "bar"},  # type: ignore[arg-type]
                model_kwargs=self.model_kwargs18,
            )

    def test_sanity_check_model_kwargs(self) -> None:
        with pytest.raises(TypeError):
            benchmark(
                self.resnet18,
                self.input_data,
                self.loss,
                model_kwargs=1,  # type: ignore[arg-type]
            )
