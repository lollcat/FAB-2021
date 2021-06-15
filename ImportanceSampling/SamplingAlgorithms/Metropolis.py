import torch

def Metropolis_transition(x, n_updates, log_p_x_func, noise_scalings):
    """
    :param x: initial x
    :param n_updates: number of metropolis updates
    :param log_p_x_func: (potentiall unnormalised target function)
    :param noise_scalings: can be sequence e.g. e.g. tensor([2.0, 1.0, 0.1])
    :return: x sample from p_x_func
    """
    for noise_scaling in noise_scalings:
        for n in range(n_updates):
            x_proposed = x + torch.randn(x.shape).to(x.device) * noise_scaling
            x_proposed_log_prob = log_p_x_func(x_proposed)
            x_prev_log_prob = log_p_x_func(x)
            acceptance_probability = torch.exp(x_proposed_log_prob - x_prev_log_prob)
            # not that sometimes this will be greater than one, corresonding to 100% probability of acceptance
            accept = (acceptance_probability > torch.rand(acceptance_probability.shape).to(x.device)).int()
            accept = accept[:, None].repeat(1, x.shape[-1])
            x = accept * x_proposed + (1 - accept) * x
    return x


if __name__ == '__main__':
    from TargetDistributions.MoG import MoG
    from FittedModels.Models.DiagonalGaussian import DiagonalGaussian
    from Utils.numerical_utils import MC_estimate_true_expectation
    from Utils.numerical_utils import quadratic_function as expectation_function
    import matplotlib.pyplot as plt

    dim = 2
    target = MoG(dim=dim)
    learnt_sampler = DiagonalGaussian(dim=dim)
    sampler_samples = learnt_sampler.sample((5000,)).cpu().detach()
    x_metropolis = Metropolis_transition(sampler_samples, n_updates=100, log_p_x_func=target.log_prob,
                                         noise_scalings=torch.tensor([1.0, 0.1]))
    plt.plot(sampler_samples[:, 0], sampler_samples[:, 1], "o")
    plt.title("sampler samples")
    plt.show()
    plt.plot(x_metropolis[:, 0], x_metropolis[:, 1], "o")
    plt.title("annealed samples")
    plt.show()
    true_samples = target.sample((5000, )).cpu().detach()
    plt.plot(true_samples[:, 0], true_samples[:, 1], "o")
    plt.title("true samples")
    plt.show()