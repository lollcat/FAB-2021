import torch.nn as nn
import torch
class FindLowPointsByDescent(nn.Module):
    def __init__(self, target_log_prob, shape, n_points=10):
        super(FindLowPointsByDescent, self).__init__()
        self.weight = nn.Parameter(torch.zeros((n_points, *shape)))
        nn.init.normal_(self.weight, mean=0.0, std=0.2)
        self.target_log_prob = target_log_prob

    def forward(self, x):
        return self.target_log_prob(self.weight)


def grad_descent_search(target_log_prob, shape=(22, 3), epochs = 10000, n_points = 10,
                        per_print=10):
    model = FindLowPointsByDescent(target_log_prob, shape, n_points=n_points)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    loss_history = []
    max_log_prob = []
    weight_history = []
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        optimizer.zero_grad()
        log_prob = model(None)
        loss = -torch.sum(log_prob)
        loss.backward(retain_graph=True) # getting OpenMM error
        optimizer.step()
        loss_history.append(loss.item())
        max_log_prob.append(log_prob.max().item())
        weight_history.append(model.weight[0][0].item())
        if epoch % per_print == 0:
            pbar.set_description(f"loss: {round(loss.item(), 2)}  "
                                 f"max log prob: {round(log_prob.max().item(), 2)}")
    plt.plot(loss_history)
    plt.title("loss history")
    plt.show()
    plt.plot(weight_history)
    plt.title("weight 0,0 history")
    plt.show()
    plt.plot(max_log_prob)
    plt.title("max log prob history")
    plt.show()
    return model
