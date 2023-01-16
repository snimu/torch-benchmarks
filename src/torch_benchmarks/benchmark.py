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
    loss: torch.nn.Module,
    *,
    model_args: list[Any] | tuple[Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    device: torch.device | str | int = "cuda",
    dtype: torch.dtype = torch.float32,
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
    :param dtype: The datatype of the inputs. Inputs and models will be cast to it.
                    Please use and explicit `torch.dtype`, not `int` or `float`.
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

    sanity_check_cuda()
    sanity_check_device(device)
    sanity_check_num_samples(num_samples)
    sanity_check_model_args(model_args)
    sanity_check_model_kwargs(model_kwargs)
    model = sanity_check_model_type(model_type, model_args, model_kwargs)
    model.to(device)
    loss.to(device)
    input_data = nested_to(input_data, device)
    model, input_data = sanity_check_and_move_to_dtype(dtype, model, input_data)
    output = sanity_check_model_compatibility_with_input_data(model, input_data)
    sanity_check_loss(loss, output)

    # Cleanup
    del model, output

    if verbose:
        print("Sanity checks passed. \n")

    memory_bytes_forward = 0.0
    memory_bytes_forward_backward = 0.0
    compute_time_forward = 0.0
    compute_time_forward_backward = 0.0

    if verbose:
        print("Performing benchmark...")

    for _ in tqdm(range(num_samples), disable=not verbose):
        memory_bytes, compute_time = check_forward(
            model_type, input_data, model_args, model_kwargs, device, dtype
        )
        memory_bytes_forward += memory_bytes
        compute_time_forward += compute_time

        memory_bytes, compute_time = check_forward_backward(
            model_type, input_data, loss, model_args, model_kwargs, device, dtype
        )
        memory_bytes_forward_backward += memory_bytes
        compute_time_forward_backward += compute_time

    memory_bytes_forward /= num_samples
    memory_bytes_forward_backward /= num_samples
    compute_time_forward /= num_samples
    compute_time_forward_backward /= num_samples

    model_statistics = ModelStatistics(
        device=device,
        dtype=dtype,
        memory_bytes_forward=memory_bytes_forward,
        memory_bytes_forward_backward=memory_bytes_forward_backward,
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


def sanity_check_cuda() -> None:
    if not torch.cuda.is_available():
        raise OSError("`benchmark` only works on CUDA.")


def sanity_check_device(device: torch.device | str | int) -> None:
    if not isinstance(device, (torch.device, str, int)):
        raise TypeError(
            f"Parameter `device` must be of type `torch.device`, "
            f"`str`, or `None`, not {type(device)}"
        )

    if (
        isinstance(device, torch.device) and device.type != torch.device("cuda").type
    ) or (isinstance(device, str) and "cuda" not in device):
        raise ValueError("`device` must be on CUDA.")

    try:
        _ = torch.cuda.get_device_properties(device)
    except AssertionError as e:
        raise ValueError(f"Invalid `device`: {device}") from e


def sanity_check_num_samples(num_samples: int) -> None:
    if num_samples is None:
        raise TypeError("Parameter `num_samples` must be an `int`, not `None`.")

    if not isinstance(num_samples, int):
        raise TypeError(
            f"Parameter `num_samples` must be of type `int`, not {type(num_samples)}"
        )

    if num_samples < 1:
        raise ValueError("Parameter `num_samples` must be greater than 1.")


def sanity_check_model_args(model_args: list[Any] | tuple[Any]) -> None:
    if not isinstance(model_args, (list, tuple)):
        raise TypeError(
            f"Parameter `model_args` must be a `list`, "
            f"a `tuple`, or `None`, not {type(model_args)}."
        )


def sanity_check_model_kwargs(model_kwargs: dict[str, Any]) -> None:
    if not isinstance(model_kwargs, dict) or not all(
        isinstance(key, str) for key in model_kwargs.keys()
    ):
        raise TypeError(
            f"Parameter `model_kwargs` must be a `dict[str, Any]` "
            f"or `None`, not {type(model_kwargs)}."
        )


def sanity_check_model_type(
    model_type: Any, model_args: list[Any] | tuple[Any], model_kwargs: dict[str, Any]
) -> Any:
    try:
        return model_type(*model_args, **model_kwargs)
    except Exception as e:
        raise RuntimeError("Model-construction failed.") from e


def sanity_check_and_move_to_dtype(
    dtype: torch.dtype, model: torch.nn.Module, input_data: Any
) -> tuple[torch.nn.Module, Any]:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(
            f"Parameter `dtype` must be of type `torch.dtype`, not {type(dtype)}"
        )

    try:
        model.to(dtype)
    except Exception as e:
        raise RuntimeError(f"`model.to(dtype={dtype})` failed.") from e

    try:
        input_data = nested_to(input_data, dtype)
    except Exception as e:
        raise RuntimeError(f"`input_data.to(dtype={dtype})` failed.") from e

    return model, input_data


def sanity_check_model_compatibility_with_input_data(
    model: torch.nn.Module, input_data: Any
) -> Any:
    try:
        return model(input_data)
    except Exception as e:
        raise RuntimeError("Model incompatible with input_data.") from e


def sanity_check_loss(loss: torch.nn.Module, output: Any) -> None:
    if not isinstance(loss, torch.nn.Module):
        raise TypeError(
            f"Parameter `loss` must be of type `torch.nn.Module`, " f"not {type(loss)}"
        )

    try:
        loss_ = loss(output, output)
    except Exception as e:
        raise RuntimeError("`loss` incompatible with model-output.") from e

    try:
        loss_.backward()
    except Exception as e:
        raise RuntimeError(
            "model cannot update parameters with output of `loss`."
        ) from e


def nested_to(inputs: Any, target: torch.dtype | torch.device | str | int) -> Any:
    """Moves all members of `inputs` to `target`."""

    if hasattr(inputs, "to") and callable(inputs.to):
        if isinstance(target, torch.dtype):
            return inputs.to(target)
        inputs.to(target)  # device
        return inputs
    if hasattr(inputs, "tensors"):
        nested_to(inputs.tensors, target)
    if not hasattr(inputs, "__getitem__") or not inputs:
        return inputs

    if isinstance(inputs, dict):
        for key, val in inputs.items():
            inputs[key] = nested_to(val, target)
    if isinstance(inputs, list):
        for i, val in enumerate(inputs):
            inputs[i] = nested_to(val, target)
    if isinstance(inputs, tuple):
        inputs = list(inputs)
        for i, val in enumerate(inputs):
            inputs[i] = nested_to(val, target)
        inputs = tuple(inputs)
    return inputs


@torch.no_grad()
def check_forward(
    model_type: Any,
    input_data: Any,
    model_args: list[Any] | tuple[Any],
    model_kwargs: dict[str, Any],
    device: torch.device | str | int,
    dtype: torch.dtype,
) -> tuple[float, float]:
    memory_usage_before = torch.cuda.max_memory_allocated(device)
    model = model_type(*model_args, **model_kwargs).to(device, dtype=dtype)
    model.eval()
    t0 = perf_counter()

    _ = model(input_data)

    compute_time = perf_counter() - t0
    memory_bytes = torch.cuda.max_memory_allocated(device) - memory_usage_before

    del model
    torch.cuda.reset_peak_memory_stats(device)

    return memory_bytes, compute_time


def check_forward_backward(
    model_type: Any,
    input_data: Any,
    loss: torch.nn.Module,
    model_args: list[Any] | tuple[Any],
    model_kwargs: dict[str, Any],
    device: torch.device | str | int,
    dtype: torch.dtype,
) -> tuple[float, float]:
    memory_usage_before = torch.cuda.max_memory_allocated(device)
    model = model_type(*model_args, **model_kwargs).to(device, dtype=dtype)
    model.train()
    t0 = perf_counter()

    output = model(input_data)
    loss_ = loss(output, output)
    loss_.backward()

    compute_time = perf_counter() - t0
    memory_bytes = torch.cuda.max_memory_allocated(device) - memory_usage_before

    del model, output, loss_
    torch.cuda.reset_peak_memory_stats(device)

    return memory_bytes, compute_time
