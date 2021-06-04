from kris import Kris
import torch

class Boss:
    def __init__(self, dim=2, total_variance=1, target_kl=5, n_auxilliary=10):
        self.dim = dim
        self.total_variance = total_variance
        self.target_kl = target_kl
        self.n_auxilliary = n_auxilliary
        self.classes = []


    def run(self, n_steps=1000, n_samples=30):
        # let's do the first one
        z = torch.tensor([-9.8221, 4.4796])
        b_k = torch.zeros(self.dim)
        sigma_k_max = 1
        current_class = Kris(dim=self.dim, b_k=b_k, previous_sigmas=torch.zeros(3), z=z,
                             total_variance=self.total_variance, target_kl=self.target_kl, sigma_k_max=sigma_k_max)
        b_k = torch.zeros(n_samples, self.dim) # for next iter
        hist_all = []
        previous_sigmas = []
        for j in range(self.n_auxilliary - 1):
            hist = []
            optimizer = torch.optim.Adam(current_class.parameters())
            for i in range(n_steps):
                optimizer.zero_grad()
                loss = current_class.loss()
                loss.backward()
                optimizer.step()
                hist.append(current_class.kl().detach())
            print(hist[-1])
            previous_sigmas.append(current_class.sigma_k.detach())
            if j == 0:
                b_k += current_class.posterior.sample((n_samples, ))
            else:
                b_k += current_class.posterior.sample()
            sigma_k_max -= current_class.sigma_k
            current_class = Kris(dim=self.dim, b_k=b_k, previous_sigmas=torch.tensor(previous_sigmas), z=z,
                                 total_variance=self.total_variance, target_kl=self.target_kl
                                 , sigma_k_max=sigma_k_max)
            hist_all.append(hist)

        previous_sigmas = list(previous_sigmas)
        previous_sigmas.append(1 - sum(previous_sigmas))
        return hist_all, previous_sigmas

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test = Boss()
    hist, sigmas = test.run()
    plt.plot(hist[-1])
    plt.show()
    plt.plot(sigmas)
    plt.show()

