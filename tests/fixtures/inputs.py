# pylint: disable=too-few-public-methods
from __future__ import annotations

from typing import Any

import torch


class ObjectWithTensors:
    """A class with tensors."""

    def __init__(self, tensors: Any) -> None:
        self.tensors = tensors


class InputsNestedTo:
    """Inputs for tests."""

    def __init__(self) -> None:
        self.tensor = torch.ones(3)
        self.nested_tensor = {
            "a": [1, 2, "bla", torch.ones(3)],
            "b": (
                [1, 2, 3, "foo"],
                ObjectWithTensors(
                    {"c": torch.randn(3), "d": torch.randn(3), "e": torch.randn(3)}
                ),
            ),
        }

    def retrieve_tensors_from_nested_tensor(self) -> list[torch.Tensor]:
        tensors: list[Any] = [  # mypy cannot handle the ObjectWithTensors...
            self.nested_tensor["a"][3],
            self.nested_tensor["b"][1].tensors["c"],  # type: ignore[attr-defined]
            self.nested_tensor["b"][1].tensors["d"],  # type: ignore[attr-defined]
            self.nested_tensor["b"][1].tensors["e"],  # type: ignore[attr-defined]
        ]

        return tensors
