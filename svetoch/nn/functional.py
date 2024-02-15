import svetoch.autograd as ag

def log_softmax(x):
    return ag.LogSoftmaxFn()(x)


def linear(x, weight, bias=None):
    return ag.LinearFn()(x, weight, bias)


def relu(x):
    return ag.ReLUFn()(x)


def conv2d(x, weight, bias=None, stride=1, padding=0):
    return ag.Conv2dFn()(x, weight, bias, stride=stride, padding=padding)


def maxpool2d(x, ksize, stride=1, padding=0):
    return ag.MaxPool2dFn()(x, ksize, stride=stride, padding=padding)


def nll_loss(x, target):
    n = len(x)
    return -x[x.am.arange(n), target.v].sum() / x.v.dtype.type(n)


def cross_entropy(x, target):
    return nll_loss(log_softmax(x), target)
