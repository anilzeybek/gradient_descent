import torch


x = torch.arange(10)
y_target = 2*x


def objective(w):
    y = x * w  # forward pass
    return ((y - y_target)**2).sum()


def derivative_of_objective(w):
    return torch.autograd.grad(objective(w), w)[0]


weight = torch.tensor([142.3], requires_grad=True)
for _ in range(100):
    weight = weight - 0.001*derivative_of_objective(weight)

print(weight)
