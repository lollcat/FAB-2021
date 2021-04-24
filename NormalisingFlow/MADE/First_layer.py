import torch
import torch.nn as nn
import torch.nn.functional as F

class FirstLayer(nn.Module):
    def __init__(self, input_dim, layer_width):
        super(FirstLayer, self).__init__()
        self.latent_to_layer = torch.nn.utils.weight_norm(
            FirstLayerMask(latent_dim=input_dim, layer_width=layer_width))

    def forward(self, x):
        x = self.latent_to_layer(x)
        return F.elu(x)

class FirstLayerMask(nn.Module):
    def __init__(self, latent_dim, layer_width):
        super(FirstLayerMask, self).__init__()
        weight = torch.Tensor(latent_dim, layer_width)
        self.weight = nn.Parameter(weight)
        nn.init.kaiming_normal_(self.weight)

        autoregressive_mask = torch.zeros(latent_dim, layer_width)
        nodes_per_latent_representation_dim = layer_width/(latent_dim - 1)  # first latent dim has node it "owns"
        for layer_node in range(layer_width):
            layer_nodes_highest_index_latent_element_dependency = int(layer_node//nodes_per_latent_representation_dim)
            autoregressive_mask[0:layer_nodes_highest_index_latent_element_dependency+1, layer_node] = 1
        self.register_buffer("autoregressive_mask", autoregressive_mask)

    def forward(self, x):
        return torch.matmul(x, self.weight*self.autoregressive_mask)

if __name__ == "__main__":
    # test first layer mask
    test_tensor = torch.tensor([[1.5, 4.7, 5]])
    first_lay = torch.nn.utils.weight_norm(FirstLayerMask(latent_dim=test_tensor.shape[1], layer_width=6))
    print(first_lay(test_tensor))

    # test first layer
    z_test_tensor = torch.tensor([[1.5, 4.7, 5]])
    first_layer = FirstLayer(input_dim=z_test_tensor.shape[1], layer_width=6)
    print(first_layer(z_test_tensor))

    # debug check
    # the NN must not be dependent on the last element of z
    assert torch.sum(first_lay.autoregressive_mask[-1, :]) == 0
    print(first_lay.autoregressive_mask)

