import abc
import torch

class BaseTargetDistribution(abc.ABC):
    @abc.abstractmethod
    def log_prob(self, x: torch.tensor) -> torch.tensor:
        """returns (unnormalised) log probability of samples x"""


