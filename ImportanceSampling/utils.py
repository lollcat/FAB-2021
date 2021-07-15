import torch
import math
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling as ImportanceSampling
from typing import List
import matplotlib.pyplot as plt


def get_expectation_estimation_with_increasing_n_points(importance_sampler: ImportanceSampling,
                                                         start: float = 1.0, stop: float = 1.0e5, n: int = 20) \
        -> (List[float], List[any]):
    log_space = torch.logspace(math.log10(start), math.log10(stop), steps=n, dtype=torch.int)

    expectation_estimations = []
    for n_samples in log_space:
        expectation, sampling_weights = importance_sampler.calculate_expectation(n_samples)
        expectation_estimations.append(expectation)
    return log_space, expectation_estimations



if __name__ == '__main__':
    from Target_distributions.Guassian_FullCov import Unnormalised_Guassian_FullCov
    from Learnt_distribution.simple_learnable_dist_model import LearntDist
    size = 3
    target_dist = Unnormalised_Guassian_FullCov(size)
    sampling_dist = LearntDist(size)
    importance_sampler = ImportanceSampling(sampling_dist, target_dist)
    log_space, expectation_estimations = get_expectation_estimation_with_increasing_n_points(importance_sampler)
    true_expectation = torch.sum(target_dist.mean)

    plt.plot(log_space, expectation_estimations)
    #plt.plot((log_space[0], log_space[-1]), (true_expectation, true_expectation))
    plt.axhline(true_expectation, c="black")
    plt.legend(["estimation", "true"])
    plt.show()
