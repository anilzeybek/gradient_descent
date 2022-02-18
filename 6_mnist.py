import torch
import torch.nn.functional as F
import numpy as np


def fetch(url):
    import requests
    import gzip
    import os
    import hashlib
    import numpy
    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]


def objective(w1, w2):
    samp = np.random.randint(0, X_train.shape[0], size=32)
    y = torch.Tensor(X_train[samp].reshape((-1, 28*28))) @ w1
    y = torch.maximum(y, torch.zeros_like(y))
    y = y @ w2
    return F.cross_entropy(y, torch.tensor(Y_train[samp]).long())


def derivative_of_objective(w1, w2):
    dW1 = torch.autograd.grad(objective(w1, w2), w1)[0]
    dW2 = torch.autograd.grad(objective(w1, w2), w2)[0]
    return dW1, dW2


weight1 = torch.rand((784, 128)) * 0.6 - 0.3
weight2 = torch.rand((128, 10)) * 0.6 - 0.3
weight1.requires_grad = True
weight2.requires_grad = True
for _ in range(10000):
    dW1, dW2 = derivative_of_objective(weight1, weight2)
    weight1 = weight1 - 1e-4 * dW1
    weight2 = weight2 - 1e-4 * dW2


y = torch.Tensor(X_test.reshape((-1, 28*28))) @ weight1
y = torch.maximum(y, torch.zeros_like(y))
y = y @ weight2
accuracy = (y.argmax(dim=1).numpy() == Y_test).mean()
print(accuracy)
