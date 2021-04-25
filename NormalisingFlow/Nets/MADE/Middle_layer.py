import torch
import torch.nn as nn
import torch.nn.functional as F

class MiddleLayer(nn.Module):
    def __init__(self, latent_dim, layer_width):
        super(MiddleLayer, self).__init__()
        self.layer_to_layer = torch.nn.utils.weight_norm(
            MiddleLayerMask(latent_dim=latent_dim, layer_width=layer_width))

    def forward(self, x):
        return self.layer_to_layer(x)

class MiddleLayerMask(nn.Module):
    def __init__(self, latent_dim, layer_width):
        super(MiddleLayerMask, self).__init__()
        weight = torch.Tensor(layer_width, layer_width)
        self.weight = nn.Parameter(weight)
        nn.init.kaiming_normal_(self.weight)

        bias = torch.Tensor(layer_width)
        self.bias = nn.Parameter(bias)
        nn.init.zeros_(self.bias)

        autoregressive_mask = torch.zeros(layer_width, layer_width)
        nodes_per_latent_representation_dim = layer_width/(latent_dim - 1)  # first latent dim has node it "owns"
        for layer_node in range(layer_width):
            layer_nodes_highest_index_latent_element_dependency = int(layer_node//nodes_per_latent_representation_dim)
            autoregressive_mask[
            0:int((layer_nodes_highest_index_latent_element_dependency+1)*nodes_per_latent_representation_dim),
            layer_node] = 1
        self.register_buffer("autoregressive_mask", autoregressive_mask)

    def forward(self, x):
        x = torch.matmul(x, self.weight*self.autoregressive_mask) + self.bias
        return F.elu(x)

if __name__ == "__main__":
    # test middle layer mask
    test_tensor = torch.tensor([[1.5, 4.7, 5, 1.5, 4.7, 5]])
    latent_dim = 4
    layer_width = 6
    lay = torch.nn.utils.weight_norm(MiddleLayerMask(latent_dim=latent_dim, layer_width=layer_width))
    print(lay(test_tensor))

    # debug checks
    # first element must only be dependent on other first element
    assert torch.sum(lay.autoregressive_mask[int(layer_width/latent_dim)+1:, 0]) == 0
    # final node must be dependent on all previous nodes
    assert torch.sum(lay.autoregressive_mask[:, -1] == 0) == 0


    # test middle layer
    layer = MiddleLayer(latent_dim=3, layer_width=6)
    print(layer(test_tensor))


