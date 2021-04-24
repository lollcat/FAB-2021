import torch
import itertools
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_3D(x, z, n, ax, title=None):
    x = x.numpy()
    z = z.numpy()
    x1 = x[:, 0].reshape(n, n)
    x2 = x[:, 1].reshape(n, n)
    z = z.reshape(n, n)
    offset = -z.max() * 2
    trisurf = ax.plot_trisurf(x1.flatten(), x2.flatten(), z.flatten(), cmap=mpl.cm.jet)
    cs = ax.contour(x1, x2, z, offset=offset, cmap=mpl.cm.jet, stride=0.5, linewidths=0.5)
    ax.set_zlim(offset, z.max())
    if title is not None:
        ax.set_title(title)
    return


def plot_distribution(distribution, range=10, n_points=100):

    x_min = -range/2
    x_max = range/2
    x_points_1D = torch.linspace(x_min, x_max, n_points)
    x_points = torch.tensor(list(itertools.product(x_points_1D, repeat=2)))
    with torch.no_grad():
        p_x = torch.exp(distribution.log_prob(x_points))
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3D(x_points, p_x, n_points, ax, title="p(x)")
    return fig

def plot_func2D(function, range=10, n_points=100):
    x_min = -range/2
    x_max = range/2
    x_points_1D = torch.linspace(x_min, x_max, n_points)
    x_points = torch.tensor(list(itertools.product(x_points_1D, repeat=2)))
    with torch.no_grad():
        f_x = function(x_points)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3D(x_points, f_x, n_points, ax, title="f(x)")
    return fig

def MC_estimate_true_expectation(distribution, expectation_function, n_samples):
    # requires the distribution to be able to be sampled from
    x_samples = distribution.sample((n_samples,))
    f_x = expectation_function(x_samples)
    return torch.mean(f_x)


if __name__ == '__main__':
    from TargetDistributions.Guassian_FullCov import Guassian_FullCov
    dist = Guassian_FullCov(dim=2)
    plot_distribution(dist)
    plt.show()