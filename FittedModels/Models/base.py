import abc
import torch

class BaseLearntDistribution(abc.ABC):

    @abc.abstractmethod
    def forward(self, batch_size: int) -> (torch.tensor, torch.tensor):
        """forward pass of the model, return samples, log_probs"""

    @abc.abstractmethod
    def sample(self, shape: (int,)) -> torch.tensor:
        """returns samples from the distribution
        currently just coded for shape to be (n,) where n is the number of samples"""

    @abc.abstractmethod
    def log_prob(self, x: torch.tensor) -> torch.tensor:
        """returns log probability of samples x"""


    def batch_log_prob(self, x: torch.tensor, inner_batch_size=int(1e5)) -> torch.tensor:
        outer_batch_size = x.shape[0]
        if outer_batch_size < inner_batch_size:
            return self.forward(outer_batch_size)
        else:
            n_batches = outer_batch_size / inner_batch_size
            assert n_batches % 1 < 1e-10
            n_batches = int(n_batches)
            log_probs = []
            for i in range(n_batches):
                log_prob_subset = self.log_prob(x[i*inner_batch_size:i+1*outer_batch_size])
                log_probs.append(log_prob_subset)
        return torch.cat(log_probs, dim=0)

    def batch_sample(self, shape: (int,), inner_batch_size=int(1e5)) -> torch.tensor:
        """
        :param outer_batch_size: desired batch size
        :param inner_batch_size: batch size used within each forward (to prevent memory issues with massive batch)
        :return: output of sample, as if outer_batch_size was used
        """
        outer_batch_size = shape[0]
        if outer_batch_size < inner_batch_size:
            return self.forward(outer_batch_size)
        else:
            n_batches = outer_batch_size / inner_batch_size
            assert n_batches % 1 < 1e-10
            n_batches = int(n_batches)
            samples = []
            for _ in range(n_batches):
                sample_subset = self.sample((inner_batch_size,))
                samples.append(sample_subset)
            return torch.cat(samples, dim=0)

    def batch_forward(self, outer_batch_size, inner_batch_size=int(1e5)):
        """
        :param outer_batch_size: desired batch size
        :param inner_batch_size: batch size used within each forward (to prevent memory issues with massive batch)
        :return: output of forward, as if outer_batch_size was used
        """
        if outer_batch_size < inner_batch_size:
            return self.forward(outer_batch_size)
        else:
            n_batches = outer_batch_size / inner_batch_size
            assert n_batches % 1 < 1e-10
            n_batches = int(n_batches)
            samples = []; log_probs = []
            for _ in range(n_batches):
                sample_subset, log_probs_subset = self.forward(inner_batch_size)
                samples.append(sample_subset)
                log_probs.append(log_probs_subset)
            return torch.cat(samples, dim=0), torch.cat(log_probs, dim=0)


