import torch


x = torch.rand(100, 3) * 10
y_target = (x.sum(axis=1) ** 2) * 3


def objective(w1, w2, w3):
    y = x @ w1
    y = torch.maximum(y, torch.zeros_like(y))
    y = y @ w2
    y = torch.maximum(y, torch.zeros_like(y))
    y = y @ w3
    return ((y - y_target)**2).sum()


def derivative_of_objective(w1, w2, w3):
    dW1 = torch.autograd.grad(objective(w1, w2, w3), w1)[0]
    dW2 = torch.autograd.grad(objective(w1, w2, w3), w2)[0]
    dW3 = torch.autograd.grad(objective(w1, w2, w3), w3)[0]
    return dW1, dW2, dW3


weight1 = torch.rand((3, 16)) * 0.6 - 0.3
weight2 = torch.rand((16, 4)) * 0.6 - 0.3
weight3 = torch.rand((4, 1)) * 0.6 - 0.3
weight1.requires_grad = True
weight2.requires_grad = True
weight3.requires_grad = True
for _ in range(1000):
    dW1, dW2, dW3 = derivative_of_objective(weight1, weight2, weight3)
    weight1 = weight1 - 1e-10 * dW1
    weight2 = weight2 - 1e-10 * dW2
    weight3 = weight3 - 1e-10 * dW3


output = torch.Tensor([4, 5, 6]) @ weight1
output = torch.maximum(output, torch.zeros_like(output))
output = output @ weight2
output = torch.maximum(output, torch.zeros_like(output))
output = output @ weight3

print(output)
