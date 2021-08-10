from AIS_train.AnnealedImportanceSampler import AnnealedImportanceSampler as base
import numpy as np

class AnnealedImportanceSampler(base):
    def __init__(self, *args, **kwargs):
        super(AnnealedImportanceSampler, self).__init__(*args, **kwargs, Beta_end=2.0)


    def setup_n_distributions(self, n_distributions, distribution_spacing="linear"):
        self.n_distributions = n_distributions
        assert self.n_distributions > 1
        if self.n_distributions == 2:
            print("running without any intermediate distributions")
            intermediate_B_space = []  # no intermediate B space
        else:
            if self.n_distributions == 3:
                print("using linear spacing as there is only 1 intermediate distribution")
                intermediate_B_space  = [0.5*self.Beta_end]  # aim half way
            else:
                if distribution_spacing == "geometric":
                    n_linspace_points = max(int(n_distributions / 5),
                                            2)  # rough heuristic, copying ratio used in example in AIS paper
                    n_geomspace_points = n_distributions - n_linspace_points
                    intermediate_B_space = list(np.linspace(0, 0.1, n_linspace_points+1)[1:-1]*self.Beta_end)\
                                           + \
                                                list(np.geomspace(0.1, 1, n_geomspace_points)*self.Beta_end)[:-1]
                elif distribution_spacing == "linear":
                    intermediate_B_space = list(np.linspace(0.0, 1.0, n_distributions)[1:-1]*self.Beta_end)
                else:
                    raise Exception(f"distribution spacing incorrectly specified: '{distribution_spacing}',"
                                    f"options are 'geometric' or 'linear'")
        self.B_space = [0.0] + intermediate_B_space + [1.0*self.Beta_end]  # we always start and end with 0 and 1

if __name__ == '__main__':
    import torch
    torch.autograd.set_detect_anomaly(True)
    from FittedModels.Models.FlowModel import FlowModel
    from TargetDistributions.MoG import MoG
    import matplotlib.pyplot as plt
    torch.manual_seed(2)
    epochs = 1000
    dim = 2
    n_samples_estimation = int(1e4)
    target = MoG(dim=dim, n_mixes=2, min_cov=1, loc_scaling=3)
    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=3.0)
    HMC_transition_operator_args = {}
    test = AnnealedImportanceSampler(sampling_distribution=learnt_sampler,
                                     target_distribution=target, n_distributions=5,
                                     transition_operator="HMC",
                                     transition_operator_kwargs=HMC_transition_operator_args)
    x_new, log_w = test.run(1000)
    x_new = x_new.cpu().detach()
    plt.plot(x_new[:, 0], (x_new[:, 1]), "o")
    plt.show()

    true_samples = target.sample((1000,)).detach().cpu()
    plt.plot(true_samples[:, 0], (true_samples[:, 1]), "o")
    plt.show()
    print(test.transition_operator_class.interesting_info())