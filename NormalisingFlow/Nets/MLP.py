import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Zero initialisation for final layer is useful for training very deep flow models https: // arxiv.org / pdf / 1807.03039.pdf
    as it gives the identity transformation
    """
    def __init__(self, input_dim, output_dim, hidden_layer_width, n_hidden_layers=1,
                 init_zeros=True):
        super(MLP, self).__init__()
        in_dim = input_dim
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(nn.Linear(in_dim, hidden_layer_width))
            in_dim = hidden_layer_width
        self.output_layer = nn.Linear(in_dim, output_dim)
        if init_zeros:
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = F.elu(hidden_layer(x))
        return self.output_layer(x)
