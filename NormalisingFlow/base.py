import abc
import torch

class BaseFlow(abc.ABC):

    @abc.abstractmethod
    def forward(self, batch_size: int) -> (torch.tensor, torch.tensor):
        """forward flow, return x and det(dy/dz)"""

    @abc.abstractmethod
    def backward(self, x: torch.tensor) -> (torch.tensor, torch.tensor):
        """computes z, and det(dz/dy)"""



