import torch
import torch.nn as nn
import itertools

class Energy(torch.nn.Module):
    """
    https://zenodo.org/record/3242635#.YNna8uhKjIW
    """
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        x = x.requires_grad_(True)
        e = self.energy(x, temperature=temperature)
        return -torch.autograd.grad(e.sum(), x)[0]


class DoubleWellEnergy(Energy, nn.Module):
    def __init__(self, dim, a=0.0, b=-4., c=1.):
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x):
        d = x[:, [0]]
        v = x[:, 1:]
        e1 = self._a * d + self._b * d.pow(2) + self._c * d.pow(4)
        e2 = 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return e1 + e2

    def log_prob(self, x):
        return torch.squeeze(-self.energy(x))

class ManyWellEnergy(DoubleWellEnergy):
    def __init__(self, dim=4, *args, **kwargs):
        assert dim % 2 == 0
        self.n_wells = dim // 2
        super(ManyWellEnergy, self).__init__(dim=2, *args, **kwargs)
        centre = 1.7
        dim_1_vals_grid = torch.meshgrid([torch.tensor([-centre, centre])for _ in range(self.n_wells)])
        dim_1_vals = torch.stack([torch.flatten(dim) for dim in dim_1_vals_grid], dim=-1)
        n_modes = 2**self.n_wells
        assert n_modes == dim_1_vals.shape[0]
        self.test_set_ = torch.zeros((n_modes, dim))
        self.test_set_[:, torch.arange(dim) % 2 == 0] = dim_1_vals


    def test_set(self, device):
        return (self.test_set_ + torch.randn_like(self.test_set_)*0.2).to(device)

    def log_prob(self, x):
        return torch.sum(
            torch.stack(
                [super(ManyWellEnergy, self).log_prob(x[:, i*2:i*2+2]) for i in range(self.n_wells)]),
            dim=0)

    def log_prob_2D(self, x):
        # for plotting, given 2D x
        return super(ManyWellEnergy, self).log_prob(x)



if __name__ == '__main__':
    from Utils.plotting_utils import plot_distribution
    from FittedModels.utils.plotting_utils import plot_samples_vs_contours_many_well
    import matplotlib.pyplot as plt
    target = ManyWellEnergy(2, a=-0.5, b=-6)
    dist = plot_distribution(target, bounds=[[-3, 3], [-3, 3]], n_points=100)
    plt.plot(target.test_set_[:, 0], target.test_set_[:, 1], marker="o", c="black", markersize=15)
    plt.show()

    dim = 6
    target = ManyWellEnergy(dim, a=-0.5, b=-6)
    log_prob = target.log_prob(torch.randn((7, dim)))
    print(log_prob)
    class test_class:
        target_dist = target
        device = "cpu"
    plot_samples_vs_contours_many_well(test_class, samples_q=target.test_set("cpu"))
    plt.show()
    print(target.log_prob(target.test_set("cpu")))
