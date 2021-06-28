import numpy as np
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

class MuellerPotential(nn.Module):
    """
    https://zenodo.org/record/3242635#.YNna8uhKjIW
    currently just for 2D
    """
    params_default = {'k' : 1.0,
                      'dim' : 2}


    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]

    def __init__(self, params=None):
        super(MuellerPotential, self).__init__()
        # set parameters
        if params is None:
            params = self.__class__.params_default
        self.params = params

        # useful variables
        self.dim = self.params['dim']

    def energy_torch(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        batchsize = x.shape[0]
        value = torch.zeros(batchsize)
        for j in range(0, 4):
            value += self.AA[j] * torch.exp(self.aa[j] * (x1 - self.XX[j])**2 +
                                         self.bb[j] * (x1 - self.XX[j]) * (x2 - self.YY[j]) +
                                         self.cc[j] * (x2 - self.YY[j])**2)
        # redundant variables
        if self.dim > 2:
            value += 0.5 * torch.sum(x[:, 2:] ** 2, dim=1)

        return self.params['k'] * value

    def log_prob(self, x):
        return torch.squeeze(-self.energy_torch(x))

    def log_prob_2D(self, x):
        # for plotting, so we can just use the double well plotter
        return self.log_prob(x)



if __name__ == '__main__':
    from Utils.plotting_utils import plot_distribution, plot_contours
    import matplotlib.pyplot as plt
    target = MuellerPotential()
    bound = 3
    plot_distribution(target, bounds=[[-bound, bound], [-bound, bound]], n_points=300)
    plt.show()
    plot_contours(target,  bounds=[[-bound, bound], [-bound, bound]], n_points_contour=300)
    plt.show()

    log_prob = target.log_prob(torch.randn((7, 2)))
    print(log_prob)

