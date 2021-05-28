import torch
import matplotlib.pyplot as plt

torch.manual_seed(5)
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
from TargetDistributions.BayesianNN import PosteriorBNN
from FittedModels.Models.FlowModel import FlowModel
from FittedModels.train import LearntDistributionManager
from FittedModels.utils import plot_history

def expectation_function(x):
    A = torch.ones((x.shape[-1], x.shape[-1]))
    return torch.einsum("bi,ij,bj->b", x, A, x)

if __name__ == '__main__':
    epochs = 500
    n_samples_estimation = int(1e3)
    target = PosteriorBNN(n_datapoints=10, x_dim=1, y_dim=1, n_hidden_layers=0, layer_width=1, simple_mode=True)
    dim = target.n_parameters
    learnt_sampler = FlowModel(x_dim=dim)
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG")  #"DReG" # "DReG"
    expectation_before, info_before = tester.estimate_expectation(n_samples_estimation, expectation_function)
    history = tester.train(epochs, batch_size=500)
    expectation, info = tester.estimate_expectation(n_samples_estimation, expectation_function)

    print(f"******  Before training ************* \n"
          f" estimate  is {expectation_before} \n"
          f"effective sample size  is {info_before['effective_sample_size']} \n"
         f"variance in weights is {torch.var(info_before['normalised_sampling_weights'])} \n"
          f"******  After training ************* \n"
          f"estimate is {expectation} \n" 
         f"effective sample size is {info['effective_sample_size']} \n"
         f"variance in weights is {torch.var(info['normalised_sampling_weights'])}")

    plt.violinplot([info['normalised_sampling_weights']])
    plt.yscale("log")
    plt.show()

    plot_history(history)
    plt.show()
