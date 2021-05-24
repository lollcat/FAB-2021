import torch
import torch.nn as nn
import torch.nn.functional as F
Notebook = False
if Notebook:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from FittedModels.train import LearntDistributionManager
from ImportanceSampling.AnnealedImportanceSampler import AnnealedImportanceSampler


class AIS_trainer(LearntDistributionManager):
    """
    Merges annealed importance sampling into the training
    """
    def __init__(self, target_distribution, fitted_model,
                 n_distributions=10, n_updates_Metropolis=5, save_for_visualisation=False, save_spacing=20,
                 loss_type="kl", alpha=2):
        self.AIS_train = AnnealedImportanceSampler(fitted_model, target_distribution,
                 n_distributions=n_distributions, n_updates_Metropolis=n_updates_Metropolis,
                                        save_for_visualisation=save_for_visualisation, save_spacing=save_spacing)

        # we could feed a different importance sampler for post training expectation estimation
        # TODO rewrite without using inheritence
        super(AIS_trainer, self).__init__(target_distribution, fitted_model, self.AIS_train,
                 loss_type, alpha)


    def train(self, epochs=100, batch_size=1000):
        epoch_per_print = max(int(epochs / 10), 1)
        history = {"loss": [],
                   "log_p_x": [],
                   "log_w": []}
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.optimizer.zero_grad()
            x_samples, log_w = self.AIS_train.run(batch_size)
            loss = self.loss(log_w)
            if torch.isnan(loss):
                raise Exception("NaN loss encountered")
            loss.z_to_x()
            self.optimizer.step()
            # save info
            log_p_x = self.target_dist.log_prob(x_samples)
            history["loss"].append(loss.item())
            history["log_p_x"].append(torch.mean(log_p_x).item())
            history["log_w"].append(torch.mean(log_w).item())
            if epoch % epoch_per_print == 0 or epoch == epochs:
                pbar.set_description(f"loss: {history['loss'][-1]}, mean log p_x {torch.mean(log_p_x)}")
        return history

    def KL_loss(self, log_w):
        kl = -log_w
        return torch.mean(kl)

    def dreg_alpha_divergence_loss(self, log_w):
        with torch.no_grad():
            w_alpha_normalised_alpha = F.softmax(self.alpha*log_w, dim=-1)
        #w_alpha_normalised_alpha[minus_inf_log_p_x_indicies] = 0  # do this to prevent nan
        #log_w[minus_inf_log_p_x_indicies] = 0  # do this to prevent nan
        return torch.sum(((1 + self.alpha) * w_alpha_normalised_alpha + self.alpha * w_alpha_normalised_alpha**2) * log_w)

if __name__ == '__main__':
    from TargetDistributions.BayesianNN import PosteriorBNN
    from FittedModels.Models.FlowModel import FlowModel
    from FittedModels.utils import plot_history
    import matplotlib.pyplot as plt

    epochs = 10
    target = PosteriorBNN(n_datapoints=10, x_dim=1, y_dim=1, n_hidden_layers=1, layer_width=1)
    dim = target.n_parameters
    learnt_sampler = FlowModel(x_dim=dim, flow_type="RealNVP")
    tester = AIS_trainer(target, learnt_sampler, loss_type="DReG", n_distributions=10, n_updates_Metropolis=3)
    history = tester.train(epochs, batch_size=200)
    plot_history(history)
    plt.show()



