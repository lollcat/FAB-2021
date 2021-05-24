import abc
import torch

class BaseFlow(abc.ABC):

    @abc.abstractmethod
    def inverse(self, batch_size: int) -> (torch.tensor, torch.tensor):
        """inverse flow, from z to x, i.e. if combined with prior, is used for sampling
        return x and det(dy/dz)"""

    @abc.abstractmethod
    def forward(self, x: torch.tensor) -> (torch.tensor, torch.tensor):
        """computes z, and det(dz/dy) given x, useful for density evaluation"""





