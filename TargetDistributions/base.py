import torch
import torch.nn as nn

class BaseTargetDistribution(nn.Module):
    def __init__(self):
        super(BaseTargetDistribution, self).__init__()

    def log_prob(self, x: torch.tensor) -> torch.tensor:
        """returns (unnormalised) log probability of samples x"""
        raise NotImplementedError

    def performance_metrics(self, samples, log_w):
        raise NotImplementedError



