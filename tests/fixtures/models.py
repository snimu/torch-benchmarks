# pylint: disable=too-few-public-methods
import torch
from torch.nn import functional as F


class SimpleModel(torch.nn.Module):
    """A very simple model."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.model(x)
        return F.softmax(x_)
