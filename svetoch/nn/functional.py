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
    # x is log-softmax output. Softmax outputs a probability distribution
    # of (0, 1). Thus log-softmax is log[(0, 1)] = (-inf, 0). Negated
    # log-softmax is (0, +inf). The correct class must have a probability
    # close to 1 at softmax output and thus a positive close to 0 value
    # at negated log-softmax output. This positive value, negated output
    # of log-softmax for a correct class, is an error to be minimized.
    # Final loss value is the sum of these errors for all input samples
    # divided by the num of samples, i.e. averaged. Averaging is used
    # so that final value does not depend on the number of samples.
    return -x[x.backend.arange(n), target.array].sum() / x.array.dtype.type(n)


def cross_entropy(x, target):
    return nll_loss(log_softmax(x), target)
