from __future__ import annotations
import torch

from mypkg_ai.utils import time_it


class Dataset:

    """
    A class to represent a dataset.

    Attributes:
        inputs (list[torch.Tensor]): The inputs of the dataset.
        targets (list[torch.Tensor]): The targets of the dataset.
    """

    @classmethod
    @time_it
    def load(cls, file_path: str) -> Dataset:
        """
        Load a dataset from a file.
        """
        instance = cls()
        instance.inputs = []
        instance.targets = []
        return instance

    @time_it
    def __init__(self) -> None:
        self.inputs = []
        self.targets = []

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.inputs)

    def save(self, file_path: str) -> None:
        """
        Save the dataset to a file.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"""Dataset(nb_samples={len(self)}):
        ..."""
