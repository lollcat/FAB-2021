import torch
import torch.nn as nn

class BaseTransitionModel(nn.Module):
    def __init__(self):
        super(BaseTransitionModel, self).__init__()

    def run(self,  x, log_p_x_func, i):
        """returns new x samples after the transiiton
        i tells us which distribution we are on, which is useful if we have diff parameters for diff transitions
        """
        raise NotImplementedError

    def anneal_step_size(self, current_epoch, anneal_period):
        """Reduces step size"""
        return self.original_step_size * max((anneal_period - current_epoch)/anneal_period, 1e-3)

    def interesting_info(self):
        # dict of any interesting information for plotting
        return {}