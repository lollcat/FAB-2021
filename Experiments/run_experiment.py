import torch
from Experiments.Configurations.base import base_config, target_config
from FittedModels.Models.FlowModel import FlowModel
from Utils import plot_func2D, MC_estimate_true_expectation, plot_distribution, expectation_function


def run_experiment(config: base_config):
    for i in range(base_config["n_runs"]):
        torch.manual_seed(i)
        learnt_sampler = FlowModel(x_dim=config["x_dim"], *base_config["flow_config"])
        target = get_target(x_dim=config["x_dim"], target_config=base_config["target_config"])
        if hasattr(target, "sample"):
            if base_config["expectation_function"] == "standard":
                from Utils import expectation_function
                true_expectation = MC_estimate_true_expectation(target, expectation_function, int(1e6))



def get_target(x_dim: int, target_config: target_config):
    if target_config["type"] == "CorrelatedGuassian":
        from TargetDistributions.Guassian_FullCov import Guassian_FullCov
        target = Guassian_FullCov(dim=x_dim, scale_covariance=1)
    elif target_config["type"] == "MoG_Simple":
        pass

    return target
