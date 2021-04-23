import abc
import torch

class BaseLearntDistribution(abc.ABC):

    @abc.abstractmethod
    def forward(self, batch_size: int) -> (torch.tensor, torch.tensor):
        """forward pass of the model"""

    @abc.abstractmethod
    def sample(self, n: int) -> torch.tensor:
        """returns n samples from the distribution"""

    @abc.abstractmethod
    def log_prob(self, x: torch.tensor) -> torch.tensor:
        """returns log probability of samples x"""
