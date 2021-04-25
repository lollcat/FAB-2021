import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_width, n_hidden_layers=3):
        super(MLP, self).__init__()
        in_dim = input_dim
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(nn.Linear(in_dim, hidden_layer_width))
            in_dim = hidden_layer_width
        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = F.elu(hidden_layer(x))
        return F.elu(self.output_layer(x))
