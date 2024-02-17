import torch
from torch import nn


def init_weights(m: nn.Module) -> None:
    """
    Initialize weights of a linear module with Xavier uniform distribution.
    If the module is not a linear module, do nothing.

    Args:
        m (nn.Module): Module to initialize weights.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class Model(nn.Module):
    """
    ...
    """
    def __init__(self):
        """Initialize the network."""
        super().__init__()
        raise NotImplementedError
        self.classifier.apply(init_weights)

    def forward(self) -> torch.Tensor:
        raise NotImplementedError
