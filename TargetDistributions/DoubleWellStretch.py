import torch
from TargetDistributions.DoubleWell import DoubleWellEnergy


class StretchManyWellEnergy(DoubleWellEnergy):
    # randomly stretch/squish some dimensions
    def __init__(self, dim=4, max_scale=10, seed=0, *args, **kwargs):
        torch.manual_seed(seed)
        self.max_scale = max_scale
        self.squish_factors = (torch.rand(size=(dim,))*max_scale + 1) # sample unoformly between 1 and 1 + max_scale
        self.squish_factors[0:2] = 1.0  # leave one pair unsquished
        assert dim % 2 == 0
        self.n_wells = dim // 2
        super(StretchManyWellEnergy, self).__init__(dim=2, *args, **kwargs)
        self.dim = dim
        self.centre = 1.7
        self.max_dim_for_all_modes = 40  # otherwise we get memory issues on huuuuge test set
        if self.dim < self.max_dim_for_all_modes:
            dim_1_vals_grid = torch.meshgrid([torch.tensor([-self.centre, self.centre])for _ in range(self.n_wells)])
            dim_1_vals = torch.stack([torch.flatten(dim) for dim in dim_1_vals_grid], dim=-1)
            n_modes = 2**self.n_wells
            assert n_modes == dim_1_vals.shape[0]
            self.test_set__ = torch.zeros((n_modes, dim))
            self.test_set__[:, torch.arange(dim) % 2 == 0] = dim_1_vals
            self.test_set__ = self.get_untransformed_x(self.test_set__)
        else:
            print("using test set containing not all modes to prevent memory issues")
        self.clamp_samples = 1/(torch.ones(dim)*self.squish_factors)*2.5

    @property
    def test_set_(self):
        if self.dim < self.max_dim_for_all_modes:
            return self.test_set__
        else:
            raise NotImplemented

    def test_set(self, device):
        return (self.test_set_ + torch.randn_like(self.test_set_)*0.05).to(device)

    def get_untransformed_x(self, x):                                                        
        return x / self.squish_factors.to(x.device)

    def get_transformed_x(self, x):
        return x * self.squish_factors.to(x.device)

    def log_prob(self, x):
        x = self.get_transformed_x(x)
        return torch.sum(
            torch.stack(
                [super(StretchManyWellEnergy, self).log_prob(x[:, i * 2:i * 2 + 2]) for i in range(self.n_wells)]),
            dim=0)

    def log_prob_2D(self, x):
        # for x in the 2D problem without any stretching and squishing
        # for plotting, given 2D x
        return super(StretchManyWellEnergy, self).log_prob(x)

    @torch.no_grad()
    def performance_metrics(self, train_class, x_samples, log_w,
                            n_batches_stat_aggregation=20):
        return {}, {} # currently don't trust energy differences as useful so do nothing here