
Notebook = False
if Notebook:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
from FittedModels.train import LearntDistributionManager
from FittedModels.utils.plotting_utils import plot_samples
from AIS_train.AnnealedImportanceSampler import AnnealedImportanceSampler
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling


class Likelihood_trainer(LearntDistributionManager):
    """
    Maximise sampler log likelihood
    """
    def __init__(self, target_distribution, fitted_model, learnt_dist_kwargs={}):
        super(Likelihood_trainer, self).__init__(target_distribution, fitted_model, VanillaImportanceSampling,
                                                 **learnt_dist_kwargs)


    def train(self, epochs=100, batch_size=1000, intermediate_plots=False,
              plotting_func=plot_samples, n_plots=3,
              KPI_batch_size=int(1e4), allow_low_support=True, allow_NaN_loss=True):
        epoch_per_print = min(max(int(epochs / 100), 1), 100)  # max 100 epoch, min 1 epoch
        epoch_per_save = max(int(epochs / 100), 1)
        if intermediate_plots is True:
            epoch_per_plot = max(int(epochs / n_plots), 1)
        history = {"loss": [],
                   "kl": [],
                   "alpha_2_divergence": [],
                   }
        pbar = tqdm(range(epochs))
        for self.current_epoch in pbar:
            self.optimizer.zero_grad()
            x_samples = self.target_dist.sample((batch_size,))
            loss = - torch.mean(self.learnt_sampling_dist.log_prob(x_samples))
            if torch.isnan(loss) or torch.isinf(loss):
                if allow_NaN_loss:
                    print("Nan loss")
                    continue
                else:
                    raise Exception("NaN loss encountered")
            loss.backward()
            self.optimizer.step()
            # save info
            history["loss"].append(loss.item())
            if (
                    self.current_epoch % epoch_per_print == 0 or self.current_epoch == epochs) and self.current_epoch > 0:
                pbar.set_description(f"loss: {np.mean(history['loss'][-epoch_per_print:])}")
            if self.current_epoch % epoch_per_save == 0 or self.current_epoch == epochs:
                history["kl"].append(self.kl_MC_estimate(KPI_batch_size))
                history["alpha_2_divergence"].append(self.alpha_divergence_MC_estimate(KPI_batch_size))
            if intermediate_plots:
                if self.current_epoch % epoch_per_plot == 0:
                    plotting_func(self, n_samples=1000,
                                  title=f"training epoch, samples from flow {self.current_epoch}")
                    # make sure plotting func has option to enter x_samples directly
                    plt.show()
        return history

if __name__ == '__main__':
    import torch
    torch.manual_seed(5)
    from ImportanceSampling.VanillaImportanceSampler import VanillaImportanceSampling
    from FittedModels.Models.FlowModel import FlowModel
    from TargetDistributions.MoG import MoG
    from FittedModels.utils.plotting_utils import plot_history
    torch.set_default_dtype(torch.float64)

    dim = 2
    epochs = int(1e3)
    n_samples_estimation = int(1e5)
    batch_size = int(200)
    flow_type = "RealNVP"  # "IAF"
    initial_flow_scaling = 5.0
    n_flow_steps = 60

    target = MoG(n_mixes=2, dim=dim)
    learnt_sampler = FlowModel(x_dim=dim, n_flow_steps=n_flow_steps,
                               scaling_factor=initial_flow_scaling, flow_type=flow_type)

    tester = Likelihood_trainer(target, learnt_sampler, learnt_dist_kwargs={"lr": 1e-3, "use_GPU":False})
    history = tester.train(epochs=epochs, batch_size=batch_size, intermediate_plots=True,
                           n_plots=5)
    plot_history(history)
    plt.show()
    plt.plot(history["loss"][100:])
    #plt.yscale("log")
    plt.show()




