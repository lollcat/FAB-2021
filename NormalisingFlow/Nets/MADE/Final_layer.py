import torch
import torch.nn as nn
import torch.nn.functional as F

class FinalLayer(nn.Module):
    def __init__(self, latent_dim, layer_width):
        super(FinalLayer, self).__init__()
        self.layer_to_m = torch.nn.utils.weight_norm(
            FinalLayerMask(latent_dim=latent_dim, layer_width=layer_width))
        self.layer_to_s = torch.nn.utils.weight_norm(
            FinalLayerMask(latent_dim=latent_dim, layer_width=layer_width))

    def forward(self, x):
        return self.layer_to_m(x), self.layer_to_s(x)

class FinalLayerMask(nn.Module):
    def __init__(self, latent_dim, layer_width):
        super(FinalLayerMask, self).__init__()
        weight = torch.Tensor(layer_width, latent_dim)
        self.weight = nn.Parameter(weight)
        nn.init.kaiming_normal_(self.weight)

        bias = torch.Tensor(latent_dim)
        self.bias = nn.Parameter(bias)
        nn.init.zeros_(self.bias)

        autoregressive_mask = torch.ones(layer_width, latent_dim)
        nodes_per_latent_representation_dim = layer_width/(latent_dim - 1)  # first latent dim has node it "owns"
        for layer_node in range(layer_width):
            # now the elements are connected only to layer nodes that weren't dependent on them
            layer_nodes_highest_index_latent_element_dependency = int(layer_node//nodes_per_latent_representation_dim)
            autoregressive_mask[layer_node, 0:layer_nodes_highest_index_latent_element_dependency+1] = 0
        self.register_buffer("autoregressive_mask", autoregressive_mask)

    def forward(self, x):
        x = torch.matmul(x, self.weight*self.autoregressive_mask) + self.bias
        return x



if __name__ == "__main__":

    test_tensor = torch.tensor([[1.5, 4.7, 5, 1.5, 4.7, 5, 8]])
    lay = torch.nn.utils.weight_norm(FinalLayerMask(latent_dim=3, layer_width=test_tensor.shape[1]))
    print(lay(test_tensor))


    layer = FinalLayer(latent_dim=4, layer_width=test_tensor.shape[1])
    print(layer(test_tensor))

    # debug checks
    # first axis must have zeros as previously all neurons were connected to it
    assert torch.sum(layer.layer_to_m.autoregressive_mask[:, 0]) == 0
    # last axis was connected to all latent elements besides itself, so must not be zeros
    assert torch.sum(layer.layer_to_m.autoregressive_mask[:, -1]) != 0
