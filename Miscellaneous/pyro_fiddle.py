import torch
import pyro
from pyro.infer import HMC, MCMC
import pyro.distributions as dist
from TargetDistributions.MoG import MoG
import matplotlib.pyplot as plt

class initialiser:
    def __init__(self, dim=2):
        self.dim = dim

    def __iter__(self):
        yield {"x": torch.randn(self.dim)}


if __name__ == '__main__':
    dim = 2
    target = MoG(dim=dim, n_mixes=5)
    def potential_func(input_dict):
        x = input_dict["x"]
        return target.log_prob(x)

    initial_params = initialiser(dim=dim)
    initial_params, = initial_params
    hmc_kernel = HMC(potential_fn=potential_func, step_size=0.0855, num_steps=4)
    mcmc = MCMC(hmc_kernel, num_samples=500, warmup_steps=10, initial_params=initial_params)
    mcmc.run()
    samples = mcmc.get_samples()['x']
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.show()

    true_samples = target.sample((500,)).cpu().detach()
    plt.scatter(true_samples[:, 0], true_samples[:, 1])
    plt.show()

