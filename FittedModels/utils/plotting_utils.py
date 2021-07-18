import torch
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Utils.plotting_utils import plot_3D

def plot_marginals(learnt_dist_manager, n_samples=1000, title=None, samples_q=None,
                   clamp_samples=10, alpha=0.2):
    if samples_q is None:
        samples_q = learnt_dist_manager.learnt_sampling_dist.sample((n_samples,))
    samples_q = torch.clamp(samples_q, -clamp_samples, clamp_samples).cpu().detach().numpy()
    fig, axs = plt.subplots(learnt_dist_manager.target_dist.dim, learnt_dist_manager.target_dist.dim,
                            figsize=(3*learnt_dist_manager.target_dist.dim, 3 * learnt_dist_manager.target_dist.dim),
                            sharex="row", sharey="row")
    for i in range(learnt_dist_manager.target_dist.dim):
        for j in range(learnt_dist_manager.target_dist.dim):
            if i != j:
                axs[i, j].plot(samples_q[:, i], samples_q[:, j], "o", alpha=alpha)
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()

def plot_samples_vs_contours_many_well(learnt_dist_manager, n_samples=1000, bounds=([-3, 3], [-3, 3]),
                                       n_points_contour=100, title=None, samples_q=None,
                                       log_prob_contour=True, clamp_samples=10, alpha=0.2):
    # when we can't sample from target distribution
    if samples_q is None:
        samples_q = learnt_dist_manager.learnt_sampling_dist.sample((n_samples,))
    samples_q = torch.clamp(samples_q, -clamp_samples, clamp_samples).cpu().detach().numpy()
    x_points_dim1 = torch.linspace(bounds[0][0], bounds[0][1], n_points_contour)
    x_points_dim2 = torch.linspace(bounds[1][0], bounds[1][1], n_points_contour)
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    with torch.no_grad():
        p_x = learnt_dist_manager.target_dist.log_prob_2D(x_points.to(learnt_dist_manager.device))
        p_x = torch.clamp_min(p_x, -1000)
        if not log_prob_contour:
            p_x = torch.exp(p_x)  # bad naming convention but makes life easy
        p_x = p_x.cpu().detach().numpy()
        p_x = p_x.reshape((n_points_contour, n_points_contour))
        x_points_dim1 = x_points[:, 0].reshape((n_points_contour, n_points_contour)).numpy()
        x_points_dim2 = x_points[:, 1].reshape((n_points_contour, n_points_contour)).numpy()
    n_plots = learnt_dist_manager.target_dist.n_wells
    fig, axs = plt.subplots(n_plots, 2, figsize=(7, 3 * n_plots), sharex="row", sharey="row")
    if len(axs.shape) == 1:  # need another axis for slicing
        axs = axs[np.newaxis, :]
    for i in range(n_plots):
        axs[i, 0].plot(samples_q[:, i*2], samples_q[:, i*2+1], "o", alpha=alpha)
        axs[i, 1].contour(x_points_dim1, x_points_dim2, p_x , levels=25)
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()






def plot_distributions(learnt_dist_manager, bounds=([-10, 10], [-10, 10]), n_points=100,
                       grid=True, log_prob=True):
    # grid samples on a grid, grid off samples from the distributions themselves
    if grid is True:
        x_points_dim1 = torch.linspace(bounds[0][0], bounds[0][1], n_points)
        x_points_dim2 = torch.linspace(bounds[1][0], bounds[1][1], n_points)
        x_points_q = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2))).to(learnt_dist_manager.device)
        x_points_p = x_points_q
        with torch.no_grad():
            log_q_x = learnt_dist_manager.learnt_sampling_dist.log_prob(x_points_q)
            log_p_x = learnt_dist_manager.target_dist.log_prob(x_points_p)
    else:
        with torch.no_grad():
            x_points_q, log_q_x = learnt_dist_manager.learnt_sampling_dist(n_points ** 2)
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
        p_x[torch.isinf(p_x) & torch.isnan(p_x)] = 0  # prevent NaN from breaking plot
    if True in torch.isinf(q_x) or True in torch.isnan(q_x):
        print("Nan or inf encountered in q(x)")
        p_x[torch.isinf(q_x) & torch.isnan(q_x)] = 0  # prevent NaN from breaking plot

    x_points_q = x_points_q.cpu()
    x_points_p = x_points_p.cpu()
    q_x = q_x.cpu()
    p_x = p_x.cpu()

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plot_3D(x_points_q, q_x, n_points, ax, title="q(x)")
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    plot_3D(x_points_p, p_x, n_points, ax, title="p(x)")


def plot_divergences(history):
    figure, axs = plt.subplots(3 if "alpha_2_divergence_over_p" in history.keys() else 2
                               , 1, figsize=(6, 10))
    axs[0].plot(history["kl"])
    axs[0].set_title("MC estimate of kl(q||p)")
    axs[1].plot(history["alpha_2_divergence"])
    axs[1].set_title("MC estimate of log alpha divergence (alpha=2)")
    if "alpha_2_divergence_over_p" in history.keys():
        axs[2].plot(history["alpha_2_divergence_over_p"])
        axs[2].set_title("MC estimate of log alpha divergence (alpha=2) using p(x) to sample")
    plt.tight_layout()


def plot_sampling_info(history):
    figure, axs = plt.subplots(2, 1, figsize=(6, 10))
    axs[0].plot(history["importance_weights_var"])
    axs[0].set_yscale("log")
    axs[0].set_title("unnormalised importance weights variance")
    axs[1].plot(history["normalised_importance_weights_var"])
    axs[1].set_yscale("log")
    axs[1].set_title("normalised importance weights variance")
    plt.tight_layout()


def plot_history(history, bounds=None, running_chunk_n=15):
    figure, axs = plt.subplots(len(history), 1, figsize=(7, 3*len(history.keys())))
    for i, key in enumerate(history):
        data = pd.Series(history[key])
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        rolling_interval = int(len(data) / running_chunk_n)
        if sum(data.isna()) > 0:
            data = data.dropna()
            print(f"NaN encountered in {key} history")
        axs[i].plot(data)
        axs[i].set_title(key)
        if bounds is not None:
            mini = max(bounds[0], data.min())
            maxi = min(bounds[1], data.max())
            axs[i].set_ylim([mini, maxi])
        if key == "alpha_divergence":
            axs[i].set_yscale("log")
    plt.tight_layout()


def plot_samples(learnt_dist_manager, n_samples=1000, title=None, samples_q=None):
    if samples_q is None:
        samples_q = learnt_dist_manager.learnt_sampling_dist.sample((n_samples,))
    samples_q = torch.clamp(samples_q, -100, 100).detach().cpu()
    samples_p = learnt_dist_manager.target_dist.sample((n_samples,)).detach().cpu()
    rows = int(learnt_dist_manager.learnt_sampling_dist.dim / 2)
    fig, axs = plt.subplots(rows, 2, sharex="all", sharey="all", figsize=(7, 3 * rows))
    for row in range(rows):
        if len(axs.shape) == 1:  # need another axis for slicing
            axs = axs[np.newaxis, :]
        axs[row, 0].scatter(samples_q[:, row], samples_q[:, row + 1], alpha=0.5)
        axs[row, 0].set_title(f"q(x) samples dim {row * 2}-{row * 2 + 1}")
        axs[row, 1].scatter(samples_p[:, row], samples_p[:, row + 1], alpha=0.5)
        axs[row, 1].set_title(f"p(x) samples dim {row * 2}-{row * 2 + 1}")
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()


if __name__ == '__main__':
    pass