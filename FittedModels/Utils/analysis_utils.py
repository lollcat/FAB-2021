import torch
import numpy as np
from tqdm import tqdm

def estimate_key_info(tester, max_n_samples=1e6, min_n_samples=10, n_runs_max=10):
    n_points_space = torch.logspace(torch.log10(torch.tensor(min_n_samples)), torch.log10(torch.tensor(max_n_samples)),
                                    int(np.log10(max_n_samples) - np.log10(min_n_samples) + 1), dtype=torch.int)

    batch_size = int(max_n_samples * n_runs_max)  # maximum n_samples, not actually batch size
    alpha_2_grads_1_list = []
    alpha_2_grads_2_list = []
    kl_DReG_grads_1_list = []
    kl_DReG_grads_2_list = []
    kl_grads_1_list = []
    kl_grads_2_list = []
    for n_samples in tqdm(n_points_space):
        n_iter = batch_size / n_samples
        assert n_iter % 1 < 1e-3
        n_iter = int(n_iter)
        alpha_2_grads_1, alpha_2_grads_2, kl_DReG_grads_1, kl_DReG_grads_2, kl_grads_1, kl_grads_2 = \
            tester.compare_loss_gradients(n_batches=n_iter, batch_size=n_samples)

        alpha_2_grads_1_list.append(alpha_2_grads_1)
        alpha_2_grads_2_list.append(alpha_2_grads_2)
        kl_DReG_grads_1_list.append(kl_DReG_grads_1)
        kl_DReG_grads_2_list.append(kl_DReG_grads_2)
        kl_grads_1_list.append(kl_grads_1)
        kl_grads_2_list.append(kl_grads_2)
    return alpha_2_grads_1_list, alpha_2_grads_2_list, kl_DReG_grads_1_list, \
           kl_DReG_grads_2_list, kl_grads_1_list, kl_grads_2_list, n_points_space.numpy()

if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from FittedModels.utils import plot_distributions, plot_samples, plot_sampling_info, plot_divergences

    torch.manual_seed(5)
    from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
    from FittedModels.train import LearntDistributionManager
    from Utils.numerical_utils import MC_estimate_true_expectation
    from Utils.numerical_utils import quadratic_function as expectation_function
    from FittedModels.Models.FlowModel import FlowModel
    from TargetDistributions.MoG import MoG

    import numpy as np

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    dim = 2
    n_samples_estimation = int(1e4)
    target = MoG(dim=dim, n_mixes=10, min_cov=0, loc_scaling=3)
    torch.manual_seed(1)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=2, scaling_factor=5.0)  # , flow_type="RealNVP")
    tester = LearntDistributionManager(target, learnt_sampler, VanillaImportanceSampling, loss_type="DReG", lr=1e-3)
    samples_fig_before = plot_samples(tester)  # this just looks at 2 dimensions

    alpha_2_grads_1_list, alpha_2_grads_2_list, kl_DReG_grads_1_list, \
    kl_DReG_grads_2_list, kl_grads_1_list, kl_grads_2_list, n_points_space  = \
        estimate_key_info(tester, max_n_samples=1e3, min_n_samples=100, n_runs_max=3)
    print(alpha_2_grads_1_list[0].shape)
    print(kl_grads_2_list[0].shape)
    print(kl_DReG_grads_1_list[0].shape)