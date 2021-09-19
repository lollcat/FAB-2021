import torch

class Monitor_NaN:
    # see  https://github.com/pytorch/pytorch/issues/15131
    def __init__(self, name=None):
        self.found_Nan = False

    def overwrite_NaN_grad(self, grad, name=None, print_=True, replace_with=0.0,
                           print_first_time_only=False):
        if True in torch.isnan(grad):
            if self.found_Nan is False and print_:
                print(f"found a NaN and overwrote it during flow gradient calculation: {name}")
                if print_first_time_only:
                    self.found_Nan = True
            grad[torch.isnan(grad) | torch.isinf(grad)] = replace_with
        return grad