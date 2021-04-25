import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipLayer(nn.Module):
    def __init__(self, latent_dim):
        super(SkipLayer, self).__init__()
        self.layer_to_layer = torch.nn.utils.weight_norm(
            SkipLayerMask(latent_dim=latent_dim))

    def forward(self, x):
        return self.layer_to_layer(x)

class SkipLayerMask(nn.Module):
    def __init__(self, latent_dim):
        super(SkipLayerMask, self).__init__()
        weight = torch.Tensor(latent_dim, latent_dim)
        self.weight = nn.Parameter(weight)
        nn.init.kaiming_normal_(self.weight)

        autoregressive_mask = torch.ones(latent_dim, latent_dim)
        autoregressive_mask = torch.triu(autoregressive_mask, diagonal=1)
        self.autoregressive_mask = nn.Parameter(autoregressive_mask, requires_grad=False)

    def forward(self, x):
        x = torch.matmul(x, self.weight*self.autoregressive_mask)
        return x