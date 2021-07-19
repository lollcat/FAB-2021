import itertools

import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

from TargetDistributions.base import BaseTargetDistribution


def plot_data_generating_process(posterior_bnn, bound=15):
    x_space = torch.linspace(-bound, bound, 20)[:, None, None].to(posterior_bnn.device)
    posterior = posterior_bnn.target.model.posterior_y_given_x(x_space)
    mean = posterior.loc.detach()
    upper_bound = (torch.squeeze(mean) + torch.squeeze(torch.sqrt(posterior.covariance_matrix) * 1.96).detach())
    lower_bound = (torch.squeeze(mean) - torch.squeeze(torch.sqrt(posterior.covariance_matrix) * 1.96).detach())
    mean = torch.squeeze(mean).cpu()
    upper_bound = torch.squeeze(upper_bound).cpu()
    lower_bound = torch.squeeze(lower_bound ).cpu()
    x_space = torch.squeeze(x_space).cpu()
    X_points, Y_points = torch.squeeze(posterior_bnn.X.cpu()), torch.squeeze(posterior_bnn.Y).cpu()

    plt.plot(X_points, Y_points, "o", label="dataset")
    plt.plot( x_space, mean, "-", c="black", label="mean")
    plt.plot( x_space, upper_bound, "--", c="black", label="1.96 std")
    plt.legend()
    plt.plot(x_space, lower_bound, "--", c="black")

def plot_fitted_model(samples_w , posterior_bnn, bound=15):
    posterior_bnn.set_model_parameters(samples_w)
    x_space = torch.linspace(-bound, bound, 20)[:, None, None].to(posterior_bnn.device)
    posterior = posterior_bnn.model.posterior_y_given_x(x_space)
    mean = torch.squeeze(posterior.loc.detach()).cpu()
    x_space = torch.squeeze(x_space).cpu()
    X_points, Y_points = torch.squeeze(posterior_bnn.X.cpu()), torch.squeeze(posterior_bnn.Y).cpu()
    plt.plot(x_space, mean, "-", c="black", alpha=0.5)
    plt.plot(X_points, Y_points, "o")
    plt.legend()



class custom_layer(nn.Module):
    # designed to compute a forward pass for a batch of weights
    def __init__(self, weight_batch_size, layer_width, input_dim, linear_activations=False, use_bias=True):
        super(custom_layer, self).__init__()
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.zeros(weight_batch_size, input_dim, layer_width))
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(weight_batch_size, layer_width))
            nn.init.normal_(self.bias, mean=0.0, std=1.0)

        if linear_activations:
            self.activation = lambda x: x
        else:
            self.activation = F.elu

    def forward(self, x):
        # first dim is batch dim, second dim is weight batch dim
        x = torch.einsum("ijk,jkl->ijl", x, self.weight)
        if self.use_bias:
            x += self.bias
        return self.activation(x)


class BNN_Fast(nn.Module):
    def __init__(self, weight_batch_size=64, x_dim=2, y_dim=2, n_hidden_layers=2, layer_width=10,
                 linear_activations=False, fixed_variance=False, use_bias=True, linear_activations_output=True):
        super(BNN_Fast, self).__init__()
        self.weight_batch_size = weight_batch_size
        self.fixed_variance = fixed_variance
        self.hidden_layers = nn.ModuleList()
        in_dim = x_dim
        for i in range(n_hidden_layers):
            assert layer_width > 0
            hidden_layer = custom_layer(weight_batch_size=weight_batch_size,
                                        layer_width=layer_width, input_dim=in_dim,
                                        linear_activations=linear_activations, use_bias=use_bias)
            self.hidden_layers.append(hidden_layer)
            in_dim = layer_width
        self.output_layer_means = custom_layer(weight_batch_size=weight_batch_size,
                                               input_dim=in_dim, layer_width=y_dim, use_bias=use_bias,
                                               linear_activations=linear_activations_output)
        if fixed_variance is False:
            self.output_layer_log_stds = custom_layer(weight_batch_size=weight_batch_size,
                                               input_dim=in_dim, layer_width=y_dim, use_bias=use_bias,
                                               linear_activations=linear_activations_output)
        else:
            self.register_buffer("log_stds", torch.ones((weight_batch_size, y_dim)))

    def forward(self, x):
        return self.sample_posterior_y_given_x(x)

    @torch.no_grad()
    def sample_posterior_y_given_x(self, x):
        posterior_distribution = self.posterior_y_given_x(x)
        return posterior_distribution.sample()

    def posterior_y_given_x(self, x):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        means = self.output_layer_means(x)
        # reparameterise to keep std resonably high, so that we have resonable density of most of y
        # but not ridiculously high or low, as this gives us nans
        if self.fixed_variance is False:
            log_std = self.output_layer_log_stds(x)/20 + 1
        else:
            log_std = self.log_stds/20 + 1
        stds = torch.exp(log_std)
        return torch.distributions.MultivariateNormal(loc=means, covariance_matrix=torch.diag_embed(stds))


    def set_parameters(self, parameter_tensor):
        # parameter tensor is weight_batch_size by n_parameters
        assert parameter_tensor.numel() / self.weight_batch_size == self.n_parameters
        new_state_dict = {}
        param_counter = 0
        # keys = list(dict(model.state_dict()).keys())
        # key = keys[1]
        new_state_dict = self.state_dict()
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                n_param = parameter.numel()/self.weight_batch_size
                assert n_param % 1 == 0
                n_param = int(n_param)
                tensor = parameter_tensor[:, param_counter:n_param + param_counter]
                tensor = tensor.reshape(parameter.shape)
                assert tensor.shape[0] == self.weight_batch_size
                new_state_dict[name] = tensor
                param_counter += n_param
        self.load_state_dict(new_state_dict)

    @property
    def n_parameters(self):
        # number of trainable parameters
        n = sum([tensor.numel() for tensor in self.parameters() if tensor.requires_grad])/self.weight_batch_size
        assert n % 1 == 0
        return int(n)

class Target(nn.Module):
    """y = f_theta(x) where theta ~ N(0,1) (theta sampled once during initialisation), and x ~ N(0,1)
    Thus a target dataset is generated by sampling x and computing y
    The goal is to use this provide x & y data that we can use to get the posterior over weights of a BNN
    """
    def __init__(self, x_dim=2, y_dim=2, n_hidden_layers=2, layer_width=10,
                 fixed_variance=False, linear_activations=False, use_bias=True, linear_activations_output=True,
                 prior_x_scaling=100.0):
        super(Target, self).__init__()
        self.x_dim = x_dim
        self.model = BNN_Fast(weight_batch_size=1, x_dim=x_dim, y_dim=y_dim, n_hidden_layers=n_hidden_layers,
                              layer_width=layer_width, fixed_variance=fixed_variance,
                        linear_activations=linear_activations, use_bias=use_bias,
                              linear_activations_output=linear_activations_output)
        self.register_buffer("prior_loc", torch.zeros(x_dim))
        self.register_buffer("prior_covariance", torch.eye(x_dim)*prior_x_scaling)
        self.register_buffer("flat_weights", self.get_flat_weights())

    @property
    def prior(self):
        return torch.distributions.multivariate_normal.MultivariateNormal(loc=self.prior_loc,
                                                                       covariance_matrix=self.prior_covariance)

    @torch.no_grad()
    def sample(self, n_points=100):
        x = self.prior.sample((n_points,))[:, None, :] # expand dim as in this case the we have a weight_batch_size=1
        y = self.model(x)
        return x, y

    def generate_nice_dataset(self, n_datapoints):
        assert self.x_dim == 1  # "only works for 1 dimensional x
        x, y = self.sample(n_datapoints*10)
        bottom_x = n_datapoints // 2
        top_x = n_datapoints - bottom_x
        indices = torch.argsort(torch.squeeze(x))
        indices_selected = torch.cat([indices[0:bottom_x], indices[-top_x:]])
        x = x[indices_selected]
        y = y[indices_selected]
        return x, y



    def get_flat_weights(self):
        # to check this, make sure that if we run self.model.set_parameters(w_flat) that nothing changes
        w_flat = torch.zeros((1, self.model.n_parameters))
        param_counter = 0
        for name, parameter in self.model.named_parameters():
            n_params = parameter.numel()
            w_flat[:, param_counter:param_counter + n_params] = torch.flatten(parameter)
            param_counter += n_params
        return w_flat




class FastPosteriorBNN(BaseTargetDistribution):
    """
     p(w | X, Y) proportional to p(w) p(Y | X, w)
     where we generate X, Y datasets using the Target class
     if we set n_hidden_layer=0, x_dim=1, y_dim=1 we should be able to visualise p(w | X, Y) in 3D
    """
    def __init__(self, n_datapoints=100, weight_batch_size=64, x_dim=2, y_dim=2, n_hidden_layers=1, layer_width=5,
                 linear_activations=False, fixed_variance=False, use_bias=True, linear_activations_output=True):
        super(FastPosteriorBNN, self).__init__()
        self.model_kwargs = {"weight_batch_size": weight_batch_size, "x_dim": x_dim, "y_dim": y_dim,
                              "n_hidden_layers": n_hidden_layers, "layer_width": layer_width,
                 "linear_activations": linear_activations, "fixed_variance": fixed_variance, "use_bias": use_bias,
                             "linear_activations_output": linear_activations_output}
        self.model = BNN_Fast(**self.model_kwargs)
        self.n_parameters = self.model.n_parameters
        self.register_buffer("prior_loc", torch.zeros(self.n_parameters))
        self.register_buffer("prior_covariance", torch.eye(self.n_parameters))
        self.target = Target(x_dim, y_dim, n_hidden_layers, layer_width,
                             linear_activations=linear_activations, fixed_variance=fixed_variance, use_bias=use_bias,
                             linear_activations_output=linear_activations_output)
        X, Y = self.target.generate_nice_dataset(n_datapoints) #sample(n_datapoints)
        self.register_buffer("X", X)
        self.register_buffer("Y", Y)
        self.batch_size_changed = False
        self.device = "cpu"  # initialised onto cpu
        self.register_buffer("equivalent_w_flat_from_symmetry", self.generate_dataset_from_symmetries())

    def test_set(self, device):
        assert device == self.device
        return self.equivalent_w_flat_from_symmetry


    @torch.no_grad()
    def generate_dataset_from_symmetries(self, named_params=None):
        if self.model_kwargs["n_hidden_layers"] == 0:
            print("no hidden layers, so no symetries")
            return self.target.flat_weights
        permutations = list(itertools.permutations(list(range(self.model_kwargs["layer_width"]))))
        n_permutations = len(permutations)
        # state dicts are just to keep track of things, but equivalent_w_flat is what we will use
        if named_params is None:
            named_params = self.target.model.named_parameters()
        equivalent_state_dicts = [self.target.model.state_dict() for _ in
                                  range(n_permutations)]
        equivalent_w_flat = torch.zeros((n_permutations, self.n_parameters))
        param_counter = 0
        for name, parameter in named_params:
            n_params = parameter.numel()
            for i, perumutation in enumerate(permutations):
                if "hidden_layers" in name:
                    if "bias" in name:
                        layer_symmetry = parameter[:, perumutation]
                    else:
                        if "0" in name: # first layer
                            layer_symmetry = parameter[:, :, perumutation]
                        else:
                            # layer to layer connection
                            layer_symmetry = parameter[:, :, perumutation][:, perumutation, :]
                    equivalent_state_dicts[i][name] = layer_symmetry
                    equivalent_w_flat[i, param_counter:param_counter + n_params] = torch.flatten(layer_symmetry)
                else:  # final layer
                    if "weight" in name:
                        layer_symmetry = parameter[:, perumutation, :]
                        equivalent_state_dicts[i][name] = layer_symmetry
                        equivalent_w_flat[i, param_counter:param_counter + n_params] = torch.flatten(layer_symmetry)
                    else: # if bias then there is no syymetry,
                        layer_symmetry = parameter
                        equivalent_state_dicts[i][name] = layer_symmetry
                        equivalent_w_flat[i, param_counter:param_counter + n_params] = torch.flatten(layer_symmetry)

            param_counter += n_params
        return equivalent_w_flat


    def to(self, device):
        self.device = device
        super(FastPosteriorBNN, self).to(device)

    def set_model_parameters(self, w):
        if self.model.weight_batch_size != w.shape[0]:
            if self.batch_size_changed is False:
                print(f"changing model batch size to {w.shape[0]} (note that this will be occuring often if this "
                      f"message comes up during training")
                self.batch_size_changed = True
            self.model_kwargs["weight_batch_size"] = w.shape[0]
            self.model = BNN_Fast(**self.model_kwargs).to(self.device)
        """p(w | X, Y) proportional to p(w) p(Y | X, w)"""
        self.model.set_parameters(w)

    def log_prob(self, w):
        if len(w.shape) == 1:
            w = w[None, :]
        self.set_model_parameters(w)
        # keys = list(dict(model.state_dict()).keys())
        # key = keys[1]
        # model.state_dict()[key] == self.model.state_dict()[key] # want these to be false
        log_p_x = self.prior_w.log_prob(w)
        # joint probability of y, we sum over the probability of each data point
        log_p_Y_given_w_X = torch.sum(self.model.posterior_y_given_x(self.X).log_prob(self.Y), 0)
        return log_p_x + log_p_Y_given_w_X

    @property
    def prior_w(self):
        return torch.distributions.multivariate_normal.MultivariateNormal(
            loc=self.prior_loc, covariance_matrix=self.prior_covariance)

def check_consistent_weight_flattening(posterior_bnn):
    set_1 = [param for param in posterior_bnn.target.model.parameters()]
    posterior_bnn.target.model.set_parameters(posterior_bnn.target.flat_weights)
    set_2 = [param for param in posterior_bnn.target.model.parameters()]
    print([s1 == s2 for s1, s2 in zip(set_1, set_2)])

if __name__ == '__main__':
    posterior_bnn = FastPosteriorBNN(n_datapoints=2, x_dim=1, y_dim=1, n_hidden_layers=0, layer_width=0
                                 , linear_activations=False, fixed_variance=True, use_bias=True,
                                     linear_activations_output=False)

    assert posterior_bnn.n_parameters == 2
    from Utils.plotting_utils import plot_distribution
    import matplotlib.pyplot as plt

    plot_distribution(posterior_bnn, n_points=100)
    plt.show()

    #""" test whole thing
    weight_batch_size = 5
    posterior_bnn = FastPosteriorBNN(weight_batch_size=weight_batch_size, n_datapoints=10, x_dim=1, y_dim=1, n_hidden_layers=2, layer_width=3,
                                     fixed_variance=True)
    plot_data_generating_process(posterior_bnn)
    plt.show()
    equivalent_w_flat = posterior_bnn.generate_dataset_from_symmetries()
    print(posterior_bnn.log_prob(equivalent_w_flat))
    print(posterior_bnn.log_prob(posterior_bnn.target.flat_weights))
    check_consistent_weight_flattening(posterior_bnn)
    print(posterior_bnn.test_set("cpu").shape)
    posterior_bnn.to("cuda")
    for _ in range(5):
        samples_w = torch.randn(weight_batch_size, posterior_bnn.model.n_parameters).to("cuda")
        print(posterior_bnn.log_prob(samples_w))
    #"""
    posterior_bnn.log_prob(torch.randn(10, posterior_bnn.model.n_parameters).to("cuda"))
    #


    """ Target Test
    test_target = Target(x_dim=2)
    x, y = test_target.sample()
    pass
    """


    """FAST BNN test
    fast_bnn = BNN_Fast(weight_batch_size=5, x_dim=2, y_dim=2)
    x = torch.zeros(1, 5, 2)
    print(fast_bnn.sample_posterior_y_given_x(x).shape)
    """
    """Layer test
    input_dim = 3
    batch_size = 4
    weight_batch_size = 5
    layer_width = 6
    x = torch.ones(batch_size, weight_batch_size, input_dim)
    layer = custom_layer(weight_batch_size=weight_batch_size, layer_width=layer_width, input_dim=input_dim)
    layer(x)
    """


