from FittedModels.run_experiment import run_experiment
from datetime import datetime

if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    problem_number = 2
    target_types = ["DoubleWell", "QuadrupleWell", "TwoModes", "MoG_2D", "MoG_4D"]
    target_type = target_types[problem_number]
    save_path_base = f"Experiment_results/{target_type}__FourthExperiment__{current_time}/"

    epochs = int(2e4)
    batch_size = 100
    n_flow_steps = 64
    initial_flow_scalings = [2.0, 2.0, 1.2]
    initial_flow_scaling = initial_flow_scalings[problem_number]
    lrs = [1e-3, 5e-4, 1e-4]
    optimizers = ["Adam"] # , "Adamax", "AdamW"]
    loss_types = ["kl", "DReG"]
    annealing_options = [True, False]
    clip_grad_norm_options = [False]
    for lr in lrs:
        for optimizer in optimizers:
            if optimizer == 'AdamW':
                weight_decay = 1e-2
            else:
                weight_decay = 1e-6
            for clip_grad_norm in clip_grad_norm_options:
                for loss_type in loss_types:
                    if loss_type == "kl":
                        for annealing in annealing_options:
                            save_path = save_path_base + f"{loss_type}/{optimizer}/lr_{lr}_clip_grad_norm_{clip_grad_norm}/anneal_{annealing}"
                            run_experiment(save_path=save_path,
                                           seed=0, target_type=target_type,
                                           flow_type="RealNVP", n_flow_steps=n_flow_steps, initial_flow_scaling=initial_flow_scaling,
                                           loss_type=loss_type, epochs=epochs, batch_size=batch_size, optimizer=optimizer, lr=lr,
                                           weight_decay=weight_decay,
                                           clip_grad_norm=clip_grad_norm, annealing=annealing,
                                           train_prior=False, train_prior_epoch=5, train_prior_lr=1e-2,
                                           n_plots=10, n_samples_estimation=int(1e6))
                    else:
                        save_path = save_path_base + f"{loss_type}/{optimizer}/lr_{lr}_clip_grad_norm_{clip_grad_norm}//"
                        run_experiment(save_path=save_path,
                                       seed=0, target_type=target_type,
                                       flow_type="RealNVP", n_flow_steps=n_flow_steps, initial_flow_scaling=initial_flow_scaling,
                                       loss_type=loss_type, epochs=epochs, batch_size=batch_size, optimizer=optimizer, lr=lr,
                                       weight_decay=weight_decay,
                                       clip_grad_norm=clip_grad_norm, annealing=False,
                                       train_prior=False, train_prior_epoch=5, train_prior_lr=1e-2,
                                       n_plots=20, n_samples_estimation=int(1e5))