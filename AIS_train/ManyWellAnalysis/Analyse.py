import matplotlib.pyplot as plt
import torch
from FittedModels.Models.FlowModel import FlowModel
from AIS_train.train_AIS import AIS_trainer
import pathlib
from TargetDistributions.DoubleWell_Utils import sample_energy, bias_uncertainty, plot_energy
from TargetDistributions.DoubleWell import ManyWellEnergy
from FittedModels.utils.model_utils import sample_and_log_w_big_batch_drop_nans



problem = "ManyWell"
dim = 4
target = ManyWellEnergy(dim=dim, a=-0.5, b=-6)
epochs = 2000
n_flow_steps = 60
n_distributions = 2 + 2
batch_size = int(1e3)
KPI_batch_size = int(1e3)
n_samples_expectation = int(batch_size*2)
experiment_name = "DoubleWell"
n_plots = 3
flow_type = "RealNVP"
HMC_transition_args = {"step_tuning_method": "p_accept"}
learnt_dist_kwargs = {"lr": 1e-3, "optimizer": "AdamW"}

if __name__ == '__main__':
    learnt_sampler = FlowModel(x_dim=dim, scaling_factor=2.0, flow_type=flow_type,
                                   n_flow_steps=n_flow_steps)
    tester = AIS_trainer(target, learnt_sampler, n_distributions=n_distributions
                         , tranistion_operator_kwargs=HMC_transition_args, transition_operator="HMC",
                         **learnt_dist_kwargs)

    from_flow = lambda n_samples: sample_and_log_w_big_batch_drop_nans(tester, n_samples=n_samples, batch_size=int(1e4),
                                                                       AIS=False)
    from_AIS = lambda n_samples: sample_and_log_w_big_batch_drop_nans(tester, n_samples=n_samples, batch_size=int(1e4),
                                                                      AIS=True)
    hist_x_flow, hists_y_flow, whist_x_flow, whists_y_flow = sample_energy(from_flow,
                                                                           n_repeat=20, n_samples=int(1e4),
                                                                           x_index=2)
    bias, std, bias_w, std_w = bias_uncertainty(target, hist_x_flow, hists_y_flow, whist_x_flow, whists_y_flow)
    fig = plot_energy(target, hist_x_flow, hists_y_flow, whist_x_flow, whists_y_flow, ylabel=True, nstd=1.0)
    plt.show()
    
    """
    from FittedModels.utils.model_utils import sample_and_log_w_big_batch_drop_nans
    x, log_w = sample_and_log_w_big_batch_drop_nans(tester, 10000, 1000, AIS = False)
    short_dict, long_dict = target.performance_metrics(x, log_w)
    print(short_dict)
    tester.learnt_sampling_dist.load_model(pathlib.Path('AIS_train/ManyWellAnalysis/4_dim_2000_epoch'))
    tester.AIS_train.transition_operator_class.load_model(pathlib.Path(
        'AIS_train/ManyWellAnalysis/4_dim_2000_epoch'))
    x, log_w = sample_and_log_w_big_batch_drop_nans(tester, 10000, 1000, AIS=False)
    short_dict, long_dict = target.performance_metrics(x, log_w)
    print(short_dict)
    """



