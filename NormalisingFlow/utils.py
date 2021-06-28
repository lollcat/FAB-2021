import torch

class Monitor_NaN:
    # see  https://github.com/pytorch/pytorch/issues/15131
    def __init__(self, name=None):
        self.found_Nan = False

    def overwrite_NaN_grad(self, grad, name=None, print_=True):
        if True in torch.isnan(grad):
            if self.found_Nan is False and print_:
                print(f"found a NaN and overwrote it during flow gradient calculation: {name}")
                self.found_Nan = True
            grad[torch.isnan(grad)] = 0
        return grad