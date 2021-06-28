import torch
import torch.nn as nn
from ImportanceSampling.SamplingAlgorithms.base import BaseTransitionModel

class Metropolis(BaseTransitionModel):
    def __init__(self, n_transitions, n_updates, step_size=1.0,
                 trainable=False, auto_adjust=True):
        """
        Designed for use in annealed importance sampler
        :param n_updates: number of metropolis updates
        :param noise_scalings: can be sequence e.g. e.g. tensor([2.0, 1.0, 0.1])
        """
        super(Metropolis, self).__init__()
        step_size = torch.tensor([step_size])
        if auto_adjust:
            assert trainable == False
        self.n_distributions = n_transitions
        self.n_updates = n_updates
        self.trainable = trainable
        self.register_buffer("step_size", step_size)
        if trainable:
            raise NotImplementedError("removed this as auto adjust is fine for low dim,"
                                      "and in high dim we do HMC anyway")
        else:
            self.register_buffer("noise_scaling_ratios", torch.linspace(3.0, 0.5, n_updates).repeat(
                (n_transitions, 1)))
            self.register_buffer("original_step_size", step_size)
        self.auto_adjust = auto_adjust
        self.target_p_accept = 0.1

    def interesting_info(self):
        interesting_dict = {}
        interesting_dict[f"noise_scaling_0_0"] = self.noise_scaling_ratios[0, 0].cpu().item()
        interesting_dict[f"noise_scaling_0_-1"] = self.noise_scaling_ratios[0, -1].cpu().item()
        return interesting_dict

    @property
    def noise_scalings(self):
        return self.noise_scaling_ratios*self.step_size

    def run(self, x, log_p_x_func, i):
        """
        :param x: initial x
        :param log_p_x_func: (potentially unnormalised target function)

        :return: x sample from p_x_func
        """
        for n in range(self.n_updates):
            x_proposed = x + torch.randn(x.shape).to(x.device) * self.noise_scalings[i, n]
            x_proposed_log_prob = log_p_x_func(x_proposed)
            x_prev_log_prob = log_p_x_func(x)
            acceptance_probability = torch.exp(x_proposed_log_prob - x_prev_log_prob)
            # not that sometimes this will be greater than one, corresonding to 100% probability of acceptance
            accept = (acceptance_probability > torch.rand(acceptance_probability.shape).to(x.device)).int()
            accept = accept[:, None].repeat(1, x.shape[-1])
            x = accept * x_proposed + (1 - accept) * x
            if self.auto_adjust:
                p_accept = torch.mean(torch.clamp_max(acceptance_probability, 1))
                if p_accept > self.target_p_accept: # to much accept
                    self.noise_scaling_ratios[i, n] = self.noise_scaling_ratios[i, n] * 1.1
                else:
                    self.noise_scaling_ratios[i, n] = self.noise_scaling_ratios[i, n] * 0.9
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
    tester = Metropolis(n_updates=10, n_transitions=3)
    x_metropolis = tester.run(sampler_samples, log_p_x_func=target.log_prob, i=0)
    x_metropolis = tester.run(sampler_samples, log_p_x_func=target.log_prob, i=2)
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

    print(tester.interesting_info())