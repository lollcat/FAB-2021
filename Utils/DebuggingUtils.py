import torch
import time

class timer:
    def __init__(self, name, print=True, divisor=1):
        self.print = print
        self.name = name
        self.divisor = divisor # for min/h/s
        self.start = time.time()
        self.end = None
        self.duration = None

    def stop(self):
        self.end = time.time()
        self.duration = (self.end - self.start)/self.divisor
        if self.print:
            print(f"\nDuration of {self.duration} for {self.name}\n")



def check_gradients(named_parameters):
    names = []
    is_nan = []
    for n, p in named_parameters:
        if p.requires_grad:
            grads = p.grad
            names.append(n)
            if grads == None:
                is_nan.append(None)
            else:
                is_nan.append(torch.sum(torch.isnan(grads)) > 0)
    return names, is_nan