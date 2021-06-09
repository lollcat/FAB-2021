import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipLayer(nn.Module):
    def __init__(self, latent_dim, weight_norm=False, init_zeros=True):
        super(SkipLayer, self).__init__()
        if weight_norm:
            self.layer_to_layer = torch.nn.utils.weight_norm(
                SkipLayerMask(latent_dim=latent_dim, init_zeros=init_zeros))
        else:
            self.layer_to_layer = SkipLayerMask(latent_dim=latent_dim, init_zeros=init_zeros)

    def forward(self, x):
        return self.layer_to_layer(x)

class SkipLayerMask(nn.Module):
    def __init__(self, latent_dim, init_zeros=True):
        super(SkipLayerMask, self).__init__()
        weight = torch.Tensor(latent_dim, latent_dim)
        self.weight = nn.Parameter(weight)
        if init_zeros is False:
            nn.init.kaiming_normal_(self.weight)
        else:
            nn.init.zeros_(self.weight)

        autoregressive_mask = torch.ones(latent_dim, latent_dim)
        autoregressive_mask = torch.triu(autoregressive_mask, diagonal=1)
        self.register_buffer("autoregressive_mask", autoregressive_mask)

    def forward(self, x):
        x = torch.matmul(x, self.weight*self.autoregressive_mask)
        return x