import torch
import torch.nn as nn

class Energy(torch.nn.Module):

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

if __name__ == '__main__':
    from Utils.plotting_utils import plot_distribution
    import matplotlib.pyplot as plt
    target = DoubleWellEnergy(2, a=-0.5, b=-6)
    dist = plot_distribution(target, bounds=[[-3, 3], [-3, 3]], n_points=100)
    plt.show()
