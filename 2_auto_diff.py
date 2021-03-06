import torch


def objective(x):
    return 2*x**2 + 16*x


def derivative_of_objective(x):
    return torch.autograd.grad(objective(x), x)[0]


curr = torch.tensor([327.52], requires_grad=True)
for _ in range(1000):
    curr = curr - 0.01*derivative_of_objective(curr)

print(curr)
