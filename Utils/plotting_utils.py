import torch
import itertools
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_samples_single_dist(distribution, n_samples = 1000):
    samples_q = distribution.sample((n_samples,)).detach()
    fig, axs = plt.subplots(1)
    axs.scatter(samples_q[:, 0], samples_q[:, 1])
    axs.set_title("q(x) samples")


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


def plot_distribution(distribution, bounds=([-10, 10], [-10, 10]), n_points=100, grid=True):
    # plot pdf using grid if grid=True,
    # otherwise based of samples, which we need if the probability density struggles in some regions
    if grid is True:
        x_points_dim1 = torch.linspace(bounds[0][0], bounds[0][1], n_points)
        x_points_dim2 = torch.linspace(bounds[1][0], bounds[1][1], n_points)
        x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    else:
        with torch.no_grad():
            x_points = distribution.sample((n_points**2,))
    with torch.no_grad():
        p_x = torch.exp(distribution.log_prob(x_points))
    if True in torch.isinf(p_x) or True in torch.isnan(p_x):
        print("Nan or inf encountered")
        p_x[torch.isinf(p_x) & torch.isnan(p_x)] = 0    # prevent NaN from breaking plot
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
    if True in torch.isinf(f_x) or True in torch.isnan(f_x):
        print("Nan or inf encountered")
        f_x[torch.isinf(f_x) & torch.isnan(f_x)] = 0  # prevent NaN from breaking plot
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3D(x_points, f_x, n_points, ax, title="f(x)")
    return fig




if __name__ == '__main__':
    from TargetDistributions.Guassian_FullCov import Guassian_FullCov
    dist = Guassian_FullCov(dim=2)
    plot_distribution(dist, grid=True, bounds=[[0, 10], [0, 10]])
    plt.show()