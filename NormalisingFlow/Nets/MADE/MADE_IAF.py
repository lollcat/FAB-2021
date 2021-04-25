import torch
import torch.nn as nn
from NormalisingFlow.Nets.MADE.First_layer import FirstLayer
from NormalisingFlow.Nets.MADE.Middle_layer import MiddleLayer
from NormalisingFlow.Nets.MADE.Final_layer import FinalLayer
from NormalisingFlow.Nets.MADE.skip_layer import SkipLayer


class MADE_IAF(nn.Module):
    """
    MADE network with at least 1 hidden layer
    Outputs m, s as is assumed to be used with IAF transform (could generlise this)
    """
    def __init__(self, x_dim, hidden_layer_width, n_hidden_layers=2):
        super(MADE_IAF, self).__init__()
        self.FirstLayer = FirstLayer(x_dim, layer_width=hidden_layer_width)
        self.MiddleLayers = nn.ModuleList()
        for i in range(n_hidden_layers):
            self.MiddleLayers.append(MiddleLayer(latent_dim=x_dim, layer_width=hidden_layer_width))
        self.FinalLayer = FinalLayer(latent_dim=x_dim, layer_width=hidden_layer_width)
        self.m_SkipLayer = SkipLayer(latent_dim=x_dim)
        self.s_SkipLayer = SkipLayer(latent_dim=x_dim)

    def forward(self, input):
        x = self.FirstLayer(input)
        for MiddleLayer in self.MiddleLayers:
            x = MiddleLayer(x)
        m, s = self.FinalLayer(x)
        m = m + self.m_SkipLayer(input)
        s = s + self.s_SkipLayer(s)
        s = s/10 + 1.5  # reparameterise to be about +1 to +2
        return m, s


if __name__ == "__main__":
    import numpy as np
    # do some checks to ensure autoregressive property
    z_test_tensor = torch.tensor([[1.5, 4.7, 5, 76]], requires_grad=True)
    autoNN = MADE_IAF(z_test_tensor.shape[1], z_test_tensor.shape[1] * 2)
    m, s = autoNN(z_test_tensor)
    gradient_w_r_t_first_element = torch.autograd.grad(m[:, 0], z_test_tensor, only_inputs=True, retain_graph=True)[0]\
        .detach().numpy()
    assert np.sum(gradient_w_r_t_first_element) == 0

    gradient_w_r_t_last_element = torch.autograd.grad(m[:, -1], z_test_tensor, only_inputs=True, retain_graph=True,)[0]\
        .detach().numpy()
    # final element must be dependent on all previous one (no zeros derivatives present)
    assert np.sum(gradient_w_r_t_last_element[:, 0:-1] == 0) == 0
    # final element of output not dependent of final element of input
    assert np.sum(gradient_w_r_t_last_element[:, -1] != 0) == 0

    jacobian = np.zeros((z_test_tensor.shape[1], m.shape[1]))
    for i in range(m.shape[1]):
        jacobian[:, i] = torch.autograd.grad(m[:, i], z_test_tensor, only_inputs=True, retain_graph=True)[0].detach().numpy()
    print(jacobian)
    assert np.allclose(jacobian, np.triu(jacobian, 1))