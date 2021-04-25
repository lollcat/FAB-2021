import torch
import itertools
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
from FittedModels.train import LearntDistributionManager
from Utils import plot_3D

def plot_history(history):
    figure, axs = plt.subplots(len(history), 1, figsize=(6, 10))
    for i, key in enumerate(history):
        axs[i].plot(history[key])
        axs[i].set_title(key)
        if key == "alpha_divergence":
            axs[i].set_yscale("log")
    return figure, axs


def plot_distributions(learnt_dist_manager: LearntDistributionManager, x_min=4, x_max=-4, n_points=100):
    x_points_1D = torch.linspace(x_min, x_max, n_points)
    x_points = torch.tensor(list(itertools.product(x_points_1D, repeat=2)))
    with torch.no_grad():
        q_x = torch.exp(learnt_dist_manager.learnt_sampling_dist.log_prob(x_points))
        p_x = torch.exp(learnt_dist_manager.target_dist.log_prob(x_points))

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plot_3D(x_points, q_x, n_points, ax, title="q(x)")
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    plot_3D(x_points, p_x, n_points, ax, title="p(x)")
    return fig


if __name__ == '__main__':
    learnt_dist_manager = LearntDistributionManager()
    plot_distributions(learnt_dist_manager)