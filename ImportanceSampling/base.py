import abc
import torch
import torch.nn.functional as F

class BaseImportanceSampler(abc.ABC):

    @abc.abstractmethod
    def calculate_expectation(self, n_samples: int, expectation_function)\
            -> (torch.tensor, dict):
        """This is the job of the importance sampler"""

    @staticmethod
    def effective_sample_size(normalised_sampling_weights):
        # effective sample size, see https://arxiv.org/abs/1602.03572
        return 1 / torch.sum(normalised_sampling_weights ** 2)

    def effective_sample_size_unnormalised_log_weights(self, unnormalised_sampling_log_weights):
        normalised_sampling_weights = F.softmax(unnormalised_sampling_log_weights, dim=-1)
        return self.effective_sample_size(normalised_sampling_weights)