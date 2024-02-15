import functools

import svetoch as svet
from . import functional as F
from . import init

class Module:
    def __init__(self):
        self._modules = []
        self.fwd_hooks, self.bwd_hooks = [], []
        self.extra_repr = ""

    def parameters(self):
        p = []
        for m in [self] + self._modules:
            if hasattr(m, "weight"):
                p.append(m.weight)
                if m.bias is not None:
                    p.append(m.bias)
        return p

    def do_grad_(self, do_grad=True):
        for p in self.parameters():
            p.do_grad_(do_grad)
        return self

    def to(self, dtype=None, device=None):
        for p in self.parameters():
            p.to_(dtype=dtype, device=device)
        return self

    def cuda(self, device=None):
        if device is None:
            device = "cuda"
        return self.to(device=device)

    def cpu(self):
        return self.to(device="cpu")

    def __call__(self, x):
        out = self.forward(x)
        if self.fwd_hooks:
            for hook in self.fwd_hooks:
                hook(self, x, out)
        if self.bwd_hooks and out.fn:
            for hook in self.bwd_hooks:
                out.fn.add_backward_hook(functools.partial(hook, self))
        return out

    def add_forward_hook(self, hook):
        self.fwd_hooks.append(hook)

    def add_backward_hook(self, hook):
        self.bwd_hooks.append(hook)

    def __getitem__(self, key):
        return self._modules[key]

    def __len__(self):
        return len(self._modules)

    def __repr__(self):
        lines = []
        for i, m in enumerate(self._modules):
            mlines = repr(m).split("\n")
            mlines[0] = "%s: %s" % (i, mlines[0])
            lines.extend(mlines)

        extra = self.extra_repr
        # currently parens are for children OR for extra
        assert not (lines and extra)
        s = self.__class__.__name__
        if extra:
            s += "(" + extra + ")"
        if lines:
            s += "(\n" + "\n".join(["  " + l for l in lines]) + "\n)"
        return s

    __str__=__repr__


class Linear(Module):
    def __init__(self, n_in, n_out, bias=True):
        super().__init__()
        self.weight = init.kaiming_normal_(svet.empty((n_in, n_out), do_grad=True))
        self.bias = None
        if bias:
            self.bias = svet.zeros((1, n_out), do_grad=True)
        self.extra_repr = "n_in=%d, n_out=%d, bias=%s" % (n_in, n_out, bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class ReLU(Module):
    def forward(self, x):
        return F.relu(x)


class Softmax(Module):
    def forward(self, x):
        e = x.exp()
        return e / e.sum(axis=1, keepdims=True)


class LogSoftmax(Module):
    def forward(self, x):
        return F.log_softmax(x)


class Seq(Module):
    def __init__(self, *modules):
        super().__init__()
        self._modules = list(modules)

    def forward(self, x):
        for m in self._modules:
            x = m(x)
        return x


class Conv2d(Module):
    def __init__(self, ch_in, ch_out, ksize, stride=1, padding=0):
        super().__init__()
        self.stride, self.padding = stride, padding
        self.weight = init.kaiming_normal_(
            svet.empty((ch_out, ch_in, ksize, ksize), do_grad=True))
        self.bias = svet.zeros(ch_out, do_grad=True)
        self.extra_repr = 'ch_in=%d, ch_out=%d, ksize=%d, stride=%d, padding=%d' % (
            ch_in, ch_out, ksize, stride, padding)

    def forward(self, x):
        return F.conv2d(x, self.w, self.b, stride=self.stride,
                      padding=self.padding)


class MaxPool2d(Module):
    def __init__(self, ksize, stride=1, padding=0):
        super().__init__()
        self.ksize, self.stride, self.padding = ksize, stride, padding

    def forward(self, x):
        return F.maxpool2d(x, self.ksize, self.stride, self.padding)
