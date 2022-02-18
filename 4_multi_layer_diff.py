import torch


x = torch.rand(10, 3) * 10
y_target = 3 * x.sum(axis=1)


def objective(w1, w2):
    y = (x @ w1) * w2  # forward pass
    return ((y - y_target)**2).sum()


def derivative_of_objective(w1, w2):
    dW1 = torch.autograd.grad(objective(w1, w2), w1)[0]
    dW2 = torch.autograd.grad(objective(w1, w2), w2)[0]
    return dW1, dW2


weight1 = torch.tensor((2.3, 6.2, 4.5), requires_grad=True)
weight2 = torch.tensor((3.5), requires_grad=True)
for _ in range(10000):
    dW1, dW2 = derivative_of_objective(weight1, weight2)
    weight1 = weight1 - 0.00001*dW1
    weight2 = weight2 - 0.00001*dW2


print((torch.Tensor([3, 4, 5]) @ weight1) * weight2)
