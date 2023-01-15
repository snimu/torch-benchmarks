from __future__ import annotations

import torch


class ModelStatistics:
    """Holds information about a model:

    *device*: The device
        (not printed out by either
        `ModelStatistics.__str__` or `ModelStatistics.__repr__`)

    *device_name*: The name of the device

    *memory_bytes_forward*:
        Memory usage of model, without gradient, forward-only
        In bytes, but printed in MB

    *memory_bytes_forward_backward*:
        Memory usage of model, with gradient, forward and backward
        In bytes, but printed in MB

    *compute_time_forward*:
        Execution time of gradient-less forward-pass of model on given device

    *compute_time_forward_backward*:
        Execution time of forward-pass followed by backward-pass
        of model on given device


    Includes `__str__`- and `__repr__`-methods (both have identical output).
    """

    def __init__(
        self,
        device: torch.device | str | int,
        memory_bytes_forward: float,
        memory_bytes_forward_backward: float,
        compute_time_forward: float,
        compute_time_forward_backward: float,
    ) -> None:
        self.device = device
        self.device_name = torch.cuda.get_device_name(device)
        self.memory_bytes_forward = memory_bytes_forward
        self.memory_bytes_forward_backward = memory_bytes_forward_backward
        self.compute_time_forward = compute_time_forward
        self.compute_time_forward_backward = compute_time_forward_backward

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        lines = [
            "Device: ",
            "Torch version: ",
            "Memory consumption (no grad, forward only): ",
            "Memory consumption (grad, forward & backward): ",
            "Compute time (no grad, forward only): ",
            "Compute time (grad, forward & backward): ",
        ]

        lines = self.fill_lines(lines)

        lines[0] += f"{self.device_name} \n"
        lines[1] += f"{torch.__version__} \n"
        lines[2] += f"{self.to_mb(self.memory_bytes_forward):.3f} MB \n"
        lines[3] += f"{self.to_mb(self.memory_bytes_forward_backward):.3f} MB \n"
        lines[4] += f"{self.compute_time_forward:.3f} sec\n"
        lines[5] += f"{self.compute_time_forward_backward:.3f} sec\n"

        max_line_len = 0
        for line in lines:
            if len(line) <= max_line_len:
                continue
            max_line_len = len(line)

        divider = "=" * max_line_len + "\n"

        return divider + "ModelStatistics \n" + divider + "".join(lines) + divider

    @staticmethod
    def to_mb(memory_bytes: float) -> float:
        return memory_bytes / 1e6

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
