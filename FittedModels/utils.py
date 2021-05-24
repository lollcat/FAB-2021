import torch
import itertools
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
from FittedModels.train import LearntDistributionManager
from Utils import plot_3D
import pandas as pd
import numpy as np

def plot_divergences(history):
    plt.figure()
    plt.plot(history["kl"])
    plt.title("MC estimate of kl(q||p)")
    plt.figure()
    plt.plot(history["alpha_2_divergence"])
    # plt.yscale("log")
    plt.title("MC estimate of log alpha divergence (alpha=2)")
    if "alpha_2_divergence_over_p" in history.keys():
        plt.figure()
        plt.plot(history["alpha_2_divergence_over_p"])
        # plt.yscale("log")
        plt.title("MC estimate of log alpha divergence (alpha=2) using p(x) to sample")

def plot_sampling_info(history):
    plt.figure()
    plt.plot(history["importance_weights_var"])
    plt.yscale("log")
    plt.title("unnormalised importance weights variance")
    plt.figure()
    plt.plot(history["normalised_importance_weights_var"])
    plt.yscale("log")
    plt.title("normalised importance weights variance")

def plot_history(history, bounds=None, running_chunk_n=30):
    figure, axs = plt.subplots(len(history), 1, figsize=(6, 10))
    for i, key in enumerate(history):
        data = pd.Series(history[key])
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        rolling_interval = int(len(data) / running_chunk_n)
        if sum(data.isna()) > 0:
            data = data.dropna()
            print(f"NaN encountered in {key} history")
        axs[i].plot(data)
        axs[i].plot(data.rolling(rolling_interval).mean())
        axs[i].set_title(key)
        if bounds is not None:
            mini = max(bounds[0], data.min())
            maxi = min(bounds[1], data.max())
            axs[i].set_ylim([mini, maxi])
        if key == "alpha_divergence":
            axs[i].set_yscale("log")
    plt.tight_layout()
    return figure, axs

def plot_samples(learnt_dist_manager: LearntDistributionManager, n_samples = 1000):
    samples_q = learnt_dist_manager.learnt_sampling_dist.sample((n_samples,))
    samples_q = torch.clamp(samples_q , -100, 100).detach()
    samples_p = learnt_dist_manager.target_dist.sample((n_samples, )).detach()
    fig, axs = plt.subplots(1, 2, sharex="all", sharey="all")
    axs[0].scatter(samples_q[:, 0], samples_q[:, 1])
    axs[0].set_title("q(x) samples")
    axs[1].scatter(samples_p[:, 0], samples_p[:, 1])
    axs[1].set_title("p(x) samples")
    return fig


def plot_distributions(learnt_dist_manager: LearntDistributionManager, bounds=([-10, 10], [-10, 10]), n_points=100,
                       grid=True, log_prob = False):
    # grid samples on a grid, grid off samples from the distributions themselves
    if grid is True:
        x_points_dim1 = torch.linspace(bounds[0][0], bounds[0][1], n_points)
        x_points_dim2 = torch.linspace(bounds[1][0], bounds[1][1], n_points)
        x_points_q = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
        x_points_p = x_points_q
        with torch.no_grad():
            log_q_x = learnt_dist_manager.learnt_sampling_dist.log_prob(x_points_q)
            log_p_x = learnt_dist_manager.target_dist.log_prob(x_points_p)
    else:
        with torch.no_grad():
            x_points_q, log_q_x = learnt_dist_manager.learnt_sampling_dist(n_points**2)
            x_points_q = torch.clamp(x_points_q, -100, 100)
            #  x_points_q = learnt_dist_manager.learnt_sampling_dist.sample((n_points ** 2,))
            # log_q_x = learnt_dist_manager.learnt_sampling_dist.log_prob(x_points_q)
            x_points_p = learnt_dist_manager.target_dist.sample((n_points ** 2,))
            log_p_x = learnt_dist_manager.target_dist.log_prob(x_points_p)
    if log_prob is False:
        q_x = torch.exp(log_q_x)
        p_x = torch.exp(log_p_x)
    else:
        q_x = log_q_x  # bad naming of variables to plot log_prob
        p_x = log_p_x
    if True in torch.isinf(p_x) or True in torch.isnan(p_x):
        print("Nan or inf encountered in p(x)")
        p_x[torch.isinf(p_x) & torch.isnan(p_x)] = 0    # prevent NaN from breaking plot
    if True in torch.isinf(q_x) or True in torch.isnan(q_x):
        print("Nan or inf encountered in q(x)")
        p_x[torch.isinf(q_x) & torch.isnan(q_x)] = 0    # prevent NaN from breaking plot
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plot_3D(x_points_q, q_x, n_points, ax, title="q(x)")
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    plot_3D(x_points_p, p_x, n_points, ax, title="p(x)")
    return fig


if __name__ == '__main__':
    learnt_dist_manager = LearntDistributionManager()
    plot_distributions(learnt_dist_manager)