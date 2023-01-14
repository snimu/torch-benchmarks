from __future__ import annotations

import sys
from time import perf_counter
from typing import Any

import torch
from tqdm import tqdm  # type: ignore[import]

from .model_statistics import ModelStatistics


def benchmark(
    model_type: Any,
    input_data: Any,
    *,
    model_args: list[Any] | tuple[Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    device: torch.device | str | int = "cuda",
    loss: torch.nn.Module | None = None,
    num_samples: int = 10,
    verbose: bool = False,
) -> ModelStatistics:
    """
    Benchmark a PyTorch model for memory-usage and compute-time.

    :param model_type: The uninitialized model.
                        Example: `torchvision.models.resnet50`.
    :param input_data: Example input-data.
    :param model_args: The arguments needed for constructing the model.
                        Example: `True`.
    :param model_kwargs: The keyword-arguments needed for constructing the model.
                        Example: `weights=torchvision.models.ResNet50_Weights`.
    :param device: The device you want your model to run on. Must be CUDA.
    :param loss: The loss-function. Optional.
    :param num_samples: The number of times the model should be measured.
    :param verbose: If `True`, prints progress info and the output of `benchmark`.
    :return: ModelStatistics object
                See torch_benchmarks/model_statistics.py for more information.
    """
    model_args = model_args if model_args is not None else []
    model_kwargs = model_kwargs if model_kwargs is not None else {}

    if verbose:
        print("\nStarted benchmark.\nPerforming sanity checks...")
    sanity_checks(
        model_type, input_data, model_args, model_kwargs, device, loss, num_samples
    )
    if verbose:
        print("Sanity checks passed. \n")

    input_data = input_data.to(device)

    memory_usage_forward = 0.0
    memory_usage_forward_backward = 0.0
    compute_time_forward = 0.0
    compute_time_forward_backward = 0.0

    if verbose:
        print("Performing benchmark...")

    for _ in tqdm(range(num_samples), disable=not verbose):
        memory_usage, compute_time = check_forward(
            model_type, input_data, model_args, model_kwargs, device
        )
        memory_usage_forward += memory_usage
        compute_time_forward += compute_time

        memory_usage, compute_time = check_forward_backward(
            model_type, input_data, model_args, model_kwargs, device, loss
        )
        memory_usage_forward_backward += memory_usage
        compute_time_forward_backward += compute_time

    memory_usage_forward /= num_samples
    memory_usage_forward_backward /= num_samples
    compute_time_forward /= num_samples
    compute_time_forward_backward /= num_samples

    model_statistics = ModelStatistics(
        model_name=model_type(*model_args, **model_kwargs).__qualname__,
        device=device,
        memory_usage_forward=memory_usage_forward,
        memory_usage_forward_backward=memory_usage_forward_backward,
        compute_time_forward=compute_time_forward,
        compute_time_forward_backward=compute_time_forward_backward,
    )

    if verbose:
        print("Done. \n")

    if not verbose:
        # pylint: disable=no-member
        verbose = not (hasattr(sys, "ps1") and sys.ps1)
    if verbose:
        print(model_statistics)

    return model_statistics


def sanity_checks(
    model_type: Any,
    input_data: Any,
    model_args: list[Any] | tuple[Any] | None,
    model_kwargs: dict[str, Any] | None,
    device: torch.device | str | int,
    loss: torch.nn.Module | None,
    num_samples: int,
) -> None:
    # cuda
    if not torch.cuda.is_available():
        raise OSError("`benchmark` only works on CUDA.")

    # device
    if not isinstance(device, (torch.device, str, int)):
        raise TypeError(
            f"Parameter `device` must be of type `torch.device`, "
            f"`str`, or `None`, not {type(device)}"
        )

    if (isinstance(device, torch.device) and device != torch.device("cuda")) or (
        isinstance(device, str) and "cuda" not in device
    ):
        raise ValueError("`device` must be on CUDA.")

    #   raises AssertionError if device not present
    _ = torch.cuda.get_device_properties(device)

    # num_samples
    if num_samples is None:
        raise TypeError("Parameter `num_samples` must be an `int`, not `None`.")

    if not isinstance(num_samples, int):
        raise TypeError(
            f"Parameter `num_samples` must be of type `int`, not {type(num_samples)}"
        )

    if num_samples < 1:
        raise ValueError("Parameter `num_samples` must be greater than 1.")

    # model_args
    if model_args is not None and not isinstance(model_args, (list, tuple)):
        raise TypeError(
            f"Parameter `model_args` must be a `list`, "
            f"a `tuple`, or `None`, not {type(model_args)}."
        )

    # model_kwargs
    if (
        model_kwargs is not None
        and not isinstance(model_kwargs, dict)
        and not all(isinstance(key, str) for key in model_kwargs.keys())
    ):
        raise TypeError(
            f"Parameter `model_kwargs` must be a `dict[str, Any]` "
            f"or `None`, not {type(model_kwargs)}."
        )

    # model_type
    try:
        model = model_type(*model_args, **model_kwargs).to(device)
    except Exception as e:
        raise RuntimeError("Model-construction failed.") from e

    # input_data
    try:
        input_data = input_data.to(device)
        output = model(input_data)
    except Exception as e:
        raise RuntimeError("Model incompatible with input_data.") from e

    # loss
    if loss is not None:
        if not isinstance(loss, torch.nn.Module):
            raise TypeError(
                f"Parameter `loss` must be of type `torch.nn.Module`, "
                f"not {type(loss)}"
            )

        try:
            output = loss(output)
        except Exception as e:
            raise RuntimeError("`loss` incompatible with model-output.") from e

        try:
            model.backward(output)
        except Exception as e:
            raise RuntimeError(
                "model cannot update parameters with output of `loss`."
            ) from e

    # Cleanup
    del model, output


@torch.no_grad()
def check_forward(
    model_type: Any,
    input_data: Any,
    model_args: list[Any] | tuple[Any],
    model_kwargs: dict[str, Any],
    device: torch.device | str | int,
) -> tuple[float, float]:
    memory_usage_before = torch.cuda.max_memory_allocated(device)
    model = model_type(*model_args, **model_kwargs).to(device)
    model.eval()
    t0 = perf_counter()

    _ = model(input_data)

    compute_time = perf_counter() - t0
    memory_usage = torch.cuda.max_memory_allocated(device) - memory_usage_before

    del model
    torch.cuda.reset_peak_memory_stats(device)

    return memory_usage, compute_time


def check_forward_backward(
    model_type: Any,
    input_data: Any,
    model_args: list[Any] | tuple[Any],
    model_kwargs: dict[str, Any],
    device: torch.device | str | int,
    loss: torch.nn.Module | None,
) -> tuple[float, float]:
    memory_usage_before = torch.cuda.max_memory_allocated(device)
    model = model_type(*model_args, **model_kwargs).to(device)
    model.train()
    t0 = perf_counter()

    output = model(input_data)
    output = output if loss is None else loss(output)
    model.backward(output)

    compute_time = perf_counter() - t0
    memory_usage = torch.cuda.max_memory_allocated(device) - memory_usage_before

    del model, output
    torch.cuda.reset_peak_memory_stats(device)

    return memory_usage, compute_time
