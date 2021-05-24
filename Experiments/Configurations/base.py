flow_config = {
    "flow_type": "IAF",  # or RealNVP
   "n_flow_steps": 3,
   "scaling_factor": 1,
   "use_exp": False,
}

target_config = {"type": "CorrelatedGuassian"}  # MoG_Simple, MoG_lumpy, DoubleWell

base_config = {"n_runs": 5,
               "flow_config": flow_config,
               "x_dim": 2,
               "target_config": target_config,
               "expectation_func": "standard"}