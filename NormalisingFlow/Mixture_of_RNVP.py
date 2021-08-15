import torch
import torch.nn as nn
from NormalisingFlow.base import BaseFlow
from NormalisingFlow.Nets.MLP import MLP
import numpy as np
"""
class MOF(BaseFlow):
    def __init__(self, flow_class, flow_args, flow_kwargs, n_mixes):
        super(MOF, self).__init__()
        self.n_mixes = n_mixes
        self.flow_components = nn.ModuleList([flow_class(*flow_args, **flow_kwargs) for _ in range(n_mixes)])
"""


class MixRealNVP(BaseFlow):
    def __init__(self, x_dim, n_mix=2, nodes_per_x=2, n_hidden_layers=1, reversed=False, use_exp=False, init_zeros=True,
                 digit_of_importance=13, t_shifter=1.0):
        super(MixRealNVP, self).__init__()
        self.use_exp = use_exp
        self.d = x_dim // 2  # elements copied from input to output
        self.D_minus_d = x_dim - self.d  # dependent on d elements
        hidden_layer_width = nodes_per_x * x_dim  # this lets us enter the layer width default argument dependent on x_dim
        self.MLPs = nn.ModuleList([MLP(self.d, self.D_minus_d*2, hidden_layer_width, n_hidden_layers=n_hidden_layers,
                       init_zeros=init_zeros) for _ in range(n_mix)])
        self.reversed = reversed
        self.n_mix = n_mix
        assert n_mix < 4  # for now to make prototype easy
        assert x_dim % 2 == 0  # for now to make prototype easy
        self.register_buffer("shift_t", torch.from_numpy(np.random.choice([-1, 1],
                                                                          size=(self.n_mix, self.d))*t_shifter))
        if x_dim >= 5:  # TODO
            self.inner_flip = True
        def inner_grabber_func(x):
            try:  # in case nans
                return int(str(x).replace(".", "")[digit_of_importance])
            except:
                return 0
        self.int_grabber_np = np.frompyfunc(inner_grabber_func, 1, 1)
        # TODO add function that lets you grab from intervidual members of the mixture (i.e. not with the index thing)

    def int_grabber(self, z_1_to_d):
        final_ints = self.int_grabber_np(z_1_to_d.detach().cpu().numpy()).astype("int")
        mixture_component_indices = final_ints % self.n_mix
        return torch.from_numpy(mixture_component_indices)


    def inverse(self, z):
        if self.reversed:
            z = z.flip(dims=(-1,))
        z_1_to_d = z[:, 0:self.d]
        z_d_plus_1_to_D = z[:, self.d:]
        x_1_to_d = z_1_to_d
        mixture_component_indices = self.int_grabber(x_1_to_d)
        sts = [MLP(z_1_to_d).split(self.D_minus_d, dim=-1) for MLP in self.MLPs]
        s_s, t_s = zip(*sts)
        s = torch.sum(torch.stack([s*(mixture_component_indices == i) for i, s in enumerate(s_s)]), dim=0)
        t = torch.sum(torch.stack([(t + self.shift_t[i])*(mixture_component_indices == i) for i, t in enumerate(t_s)]),
                      dim=0)
        # there should be nice way to slice, for now let's be hacky
        #sts = [MLP(z_1_to_d)[None, :, :].split(self.D_minus_d, dim=-1) for MLP in self.MLPs]
        #s_s = torch.cat(s_s)
        #t_s = torch.cat(t_s)
        #s = s_s[mixture_component_indices]
        #t = t_s[mixture_component_indices]
        if self.use_exp:
            raise NotImplementedError
        else:
            sigma = torch.sigmoid(s)*2 # reparameterise to start at 1
            x_d_plus_1_to_D = (z_d_plus_1_to_D - t) / sigma
            log_determinant = -torch.sum(torch.log(sigma), dim=-1)
        x = torch.cat([x_1_to_d, x_d_plus_1_to_D], dim=-1)
        if self.reversed:
            x = x.flip(dims=(-1,))
        return x, log_determinant

    def forward(self, x):
        """return z and log | dx / dz |"""
        if self.reversed:
            x = x.flip(dims=(-1,))
        x_1_to_d = x[:, 0:self.d]
        x_d_plus_1_to_D = x[:, self.d:]
        z_1_to_d = x_1_to_d
        mixture_component_indices = self.int_grabber(x_1_to_d)
        sts = [MLP(z_1_to_d).split(self.D_minus_d, dim=-1) for MLP in self.MLPs]
        s_s, t_s = zip(*sts)
        s = torch.sum(torch.stack([s * (mixture_component_indices == i) for i, s in enumerate(s_s)]), dim=0)
        t = torch.sum(
            torch.stack([(t + self.shift_t[i]) * (mixture_component_indices == i) for i, t in enumerate(t_s)]),
            dim=0)
        if self.use_exp:
            raise NotImplementedError
        else:
            sigma = torch.sigmoid(s)*2 # reparameterise to start at 1
            z_d_plus_1_to_D = x_d_plus_1_to_D * sigma + t
            log_determinant = torch.sum(torch.log(sigma), dim=-1)

        z = torch.cat([z_1_to_d, z_d_plus_1_to_D], dim=-1)
        if self.reversed:
            z = z.flip(dims=(-1,))
        return z, log_determinant

    def inverse_single_mix(self, z, mix):
        if self.reversed:
            z = z.flip(dims=(-1,))
        z_1_to_d = z[:, 0:self.d]
        z_d_plus_1_to_D = z[:, self.d:]
        x_1_to_d = z_1_to_d
        s, t = self.MLPs[mix](z_1_to_d).split(self.D_minus_d, dim=-1)
        t = t + self.shift_t[mix]
        if self.use_exp:
            raise NotImplementedError
        else:
            sigma = torch.sigmoid(s)*2 # reparameterise to start at 1
            x_d_plus_1_to_D = (z_d_plus_1_to_D - t) / sigma
            log_determinant = -torch.sum(torch.log(sigma), dim=-1)
        x = torch.cat([x_1_to_d, x_d_plus_1_to_D], dim=-1)
        if self.reversed:
            x = x.flip(dims=(-1,))
        return x, log_determinant

    def forward_single_mix(self, x, mix):
        """return z and log | dx / dz |"""
        if self.reversed:
            x = x.flip(dims=(-1,))
        x_1_to_d = x[:, 0:self.d]
        x_d_plus_1_to_D = x[:, self.d:]
        z_1_to_d = x_1_to_d
        s, t = self.MLPs[mix](z_1_to_d).split(self.D_minus_d, dim=-1)
        t = t + self.shift_t[mix]
        if self.use_exp:
            raise NotImplementedError
        else:
            sigma = torch.sigmoid(s)*2 # reparameterise to start at 1
            z_d_plus_1_to_D = x_d_plus_1_to_D * sigma + t
            log_determinant = torch.sum(torch.log(sigma), dim=-1)

        z = torch.cat([z_1_to_d, z_d_plus_1_to_D], dim=-1)
        if self.reversed:
            z = z.flip(dims=(-1,))
        return z, log_determinant

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    dim = 4
    n_steps = 3
    x = torch.randn((50, dim))*2
    layers = [MixRealNVP(x_dim=dim, reversed=i % 2 == 0, init_zeros=False) for i in range(n_steps)]
    log_det_sum = 0
    x_forward = x
    for i in range(n_steps):
        x_forward, log_determinant_forward = layers[i].inverse(x_forward)
        log_det_sum += log_determinant_forward

    x_backwards =  x_forward
    log_det_back_sum = 0
    for j in reversed(range(n_steps)):
        x_backwards, log_determinant_backward = layers[j].forward(x_backwards)
        log_det_back_sum += log_determinant_backward
    assert torch.max(torch.abs(x_backwards - x)) < 1e-6
    assert torch.max(torch.abs(log_det_sum + log_det_back_sum)) < 1e-6
    print(torch.sum(torch.abs(x_backwards - x)))
    print(torch.sum(torch.abs(log_det_sum + log_det_back_sum)))