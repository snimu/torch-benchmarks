from __future__ import annotations

import torch


class ModelStatistics:
    """Holds information about a model:


    *model_name*: The name of the model

    *device*: The device
        (not printed out by either
        `ModelStatistics.__str__` or `ModelStatistics.__repr__`)

    *device_name*: The name of the device

    *memory_usage_forward*:
        Memory usage of model, without gradient, forward-only

    *memory_usage_forward_backward*:
        Memory usage of model, with gradient, forward and backward

    *compute_time_forward*:
        Execution time of gradient-less forward-pass of model on given device

    *compute_time_forward_backward*:
        Execution time of forward-pass followed by backward-pass
        of model on given device


    Includes `__str__`- and `__repr__`-methods (both have identical output).
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device | str | int,
        memory_usage_forward: float,
        memory_usage_forward_backward: float,
        compute_time_forward: float,
        compute_time_forward_backward: float,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.device_name = torch.cuda.get_device_name(device)
        self.memory_usage_forward = memory_usage_forward
        self.memory_usage_forward_backward = memory_usage_forward_backward
        self.compute_time_forward = compute_time_forward
        self.compute_time_forward_backward = compute_time_forward_backward

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        lines = [
            model := "Model:",
            device := "Device:",
            mem_forward := "Memory consumption (no grad, forward only):",
            mem_full := "Memory consumption (grad, forward & backward): ",
            compute_forward := "Compute time (no grad, forward only): ",
            compute_forward_backward := "Compute time (grad, forward & backward): ",
        ]

        lines = self.fill_lines(lines)

        model += f"{self.model_name} \n"
        device += f"{self.device_name} \n"
        mem_forward += f"{self.to_mb(self.memory_usage_forward)} MB \n"
        mem_full += f"{self.to_mb(self.memory_usage_forward_backward)} MB \n"
        compute_forward += f"{self.compute_time_forward} sec\n"
        compute_forward_backward += f"{self.compute_time_forward_backward} sec\n"

        max_line_len = 0
        for line in lines:
            if len(line) <= max_line_len:
                continue
            max_line_len = len(line)

        divider = "=" * max_line_len + "\n"

        return divider + "ModelStatistics" + divider + "".join(lines) + divider

    @staticmethod
    def to_mb(memory_bytes: float) -> int:
        return int(memory_bytes / 1e6)

    @staticmethod
    def fill_lines(lines: list[str]) -> list[str]:
        """Fill the lines to the same length with '.'"""
        max_line_len = 0
        for line in lines:
            if len(line) <= max_line_len:
                continue
            max_line_len = len(line)

        for i, line in enumerate(lines):
            dots = "." * (max_line_len - len(line) + 4)
            lines[i] = line + dots

        return lines
