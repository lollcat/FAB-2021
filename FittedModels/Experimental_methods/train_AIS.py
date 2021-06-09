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
                 loss_type="kl", noise_scaling=1.0, alpha=2, *args, **kwargs):
        assert loss_type == "kl" # haven't got alternatives to this working
        self.AIS_train = AnnealedImportanceSampler(fitted_model, target_distribution,
                 n_distributions=n_distributions, n_updates_Metropolis=n_updates_Metropolis,
                                        save_for_visualisation=save_for_visualisation, save_spacing=save_spacing,
                                                   noise_scaling=noise_scaling)

        # we could feed a different importance sampler for post training expectation estimation
        # TODO rewrite without using inheritence
        super(AIS_trainer, self).__init__(target_distribution, fitted_model, self.AIS_train,
                 loss_type, alpha, *args, **kwargs)


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
            loss.backward()
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


if __name__ == '__main__':
    from FittedModels.Models.FlowModel import FlowModel
    from FittedModels.Utils.plotting_utils import plot_history
    import matplotlib.pyplot as plt
    import torch
    from TargetDistributions.MoG import MoG
    from Utils.plotting_utils import plot_distribution
    from Utils.numerical_utils import MC_estimate_true_expectation
    from Utils.numerical_utils import quadratic_function as expectation_function
    from FittedModels.Utils.plotting_utils import plot_samples

    torch.manual_seed(2)
    epochs = 1000
    dim = 2
    n_samples_estimation = int(1e4)
    target = MoG(dim=dim, n_mixes=2, min_cov=1, loc_scaling=3)
    true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e5))
    fig = plot_distribution(target, bounds=[[-30, 20], [-20, 20]])
    plt.show()
    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=3.0)  # , flow_type="RealNVP")
    tester = AIS_trainer(target, learnt_sampler, n_distributions=3, n_updates_Metropolis=5,
                         lr=1e-2, noise_scaling=2.0)
    plot_samples(tester)
    plt.show()
    with torch.no_grad():
        expectation, info_dict = tester.AIS_train.calculate_expectation(1000,
                                                                        expectation_function=expectation_function)
    print(f"true expectation is {true_expectation}, estimated expectation is {expectation}")
    print(
        f"ESS is {info_dict['effective_sample_size'] / 100}, "
        f"var is {torch.var(info_dict['normalised_sampling_weights'])}")

    plt.figure()
    learnt_dist_samples = learnt_sampler.sample((1000,)).cpu().detach()
    plt.scatter(learnt_dist_samples[:, 0], learnt_dist_samples[:, 1])
    plt.title("approximating distribution samples")
    plt.show()
    plt.figure()
    plt.scatter(info_dict["samples"][:, 0].cpu(), info_dict["samples"][:, 1].cpu())
    plt.title("annealed samples")
    plt.show()
    plt.figure()
    true_samples = target.sample((1000,)).cpu().detach()
    plt.scatter(true_samples[:, 0], true_samples[:, 1])
    plt.title("true samples")
    plt.show()

    history = tester.train(epochs, batch_size=100)
    plot_history(history)
    plt.show()
    plot_samples(tester)
    plt.show()





    """
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
    """


