import abc
import torch

class BaseImportanceSampler(abc.ABC):

    @abc.abstractmethod
    def calculate_expectation(self, n_samples: int, expectation_function)\
            -> (torch.tensor, dict):
        """This is the job of the importance sampler"""

    @staticmethod
    def effective_sample_size(normalised_sampling_weights):
        # effective sample size, see https://arxiv.org/abs/1602.03572
        return 1 / torch.sum(normalised_sampling_weights ** 2)