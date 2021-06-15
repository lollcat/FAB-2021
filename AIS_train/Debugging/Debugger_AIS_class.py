from train_AIS import AIS_trainer
Notebook = False
if Notebook:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
from FittedModels.utils.plotting_utils import plot_samples
import torch

from Utils.DebuggingUtils import timer

class Debugger(AIS_trainer):
    def __init__(self, *args, **kwargs):
        super(Debugger, self).__init__(*args, **kwargs)

    def train(self, epochs=100, batch_size=1000, intermediate_plots=False,
              plotting_func=plot_samples, n_plots=10,
              KPI_batch_size=int(1e4)):
        epoch_per_print = max(int(epochs / 10), 1)
        epoch_per_save = max(int(epochs / 100), 1)
        if "DReG" in self.loss_type and self.k is None:
            self.k = batch_size
        if intermediate_plots is True:
            epoch_per_plot = max(int(epochs / n_plots), 1)
        history = {"loss": [],
                   "log_p_x_after_AIS": [],
                   "log_w": []}
        if self.train_AIS_params is True:
            history.update({"noise_scaling": []})
        history.update({
           "kl": [],
           "alpha_2_divergence": []
        })
        pbar = tqdm(range(epochs))
        for self.current_epoch in pbar:
            self.optimizer.zero_grad()
            AIS_time = timer(name="AIS")
            x_samples, log_w = self.AIS_train.run(batch_size)
            AIS_time.stop()
            loss_time = timer(name="loss")
            loss = self.loss(log_w)
            loss_time.stop()
            if torch.isnan(loss) or torch.isinf(loss):
                raise Exception("NaN loss encountered")
            backprop_time = timer(name="backprop")
            loss.backward()
            self.optimizer.step()

            if self.train_AIS_params:
                self.noise_optimizer.step()
            backprop_time.stop()

            logging_time = timer(name="loggin")
            # save info
            log_p_x = self.target_dist.log_prob(x_samples)
            history["loss"].append(loss.item())
            history["log_p_x_after_AIS"].append(torch.mean(log_p_x).item())
            history["log_w"].append(torch.mean(log_w).item())
            if self.train_AIS_params is True:
                history["noise_scaling"].append(self.AIS_train.step_size.item())
            if self.current_epoch % epoch_per_print == 0 or self.current_epoch == epochs:
                pbar.set_description(f"loss: {history['loss'][-1]}, mean log p_x {torch.mean(log_p_x)}")
            if self.current_epoch % epoch_per_save == 0 or self.current_epoch == epochs:
                history["kl"].append(self.kl_MC_estimate(KPI_batch_size))
                history["alpha_2_divergence"].append(self.alpha_divergence_MC_estimate(KPI_batch_size))
            if intermediate_plots:
                if self.current_epoch % epoch_per_plot == 0:
                    plotting_func(self, n_samples=1000, title=f"training epoch, samples from flow {self.current_epoch}")
                    rows = int(self.learnt_sampling_dist.dim / 2)
                    fig, axs = plt.subplots(rows, sharex="all", sharey="all", figsize=(7, 3 * rows))
                    x_samples = x_samples.cpu().detach()  # for plotting
                    for row in range(rows):
                        if rows == 1:
                            ax = axs
                        else:
                            ax = axs[row]
                        if row == 0:
                            ax.set_title("plot of samples")
                        ax.scatter(x_samples[:, row], x_samples[:, row + 1])
                        ax.set_title(f"q(x) samples after AIS dim {row * 2}-{row * 2 + 1}")
                    plt.show()
            logging_time.stop()
        return history

if __name__ == '__main__':
    from FittedModels.Models.FlowModel import FlowModel
    import matplotlib.pyplot as plt
    from TargetDistributions.MoG import MoG
    from FittedModels.utils.plotting_utils import plot_samples

    torch.manual_seed(2)
    epochs = 5
    dim = 2
    n_samples_estimation = int(1e4)
    target = MoG(dim=dim, n_mixes=2, min_cov=1, loc_scaling=3)
    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=3.0)  # , flow_type="RealNVP")
    tester = Debugger(target, learnt_sampler, n_distributions=3, n_steps_transition_operator=3,
                         step_size=1.0, train_AIS_params=True, loss_type="kl",
                         transition_operator="HMC")
    history = tester.train(epochs, batch_size=1000, intermediate_plots=True)