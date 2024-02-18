import numpy

import svetoch as svet
import svetoch.autograd as ag


float32 = numpy.float32
float64 = numpy.float64
int64 = numpy.int64


def strip(t):
    return t.array if isinstance(t, Tensor) else t


class Tensor:
    def __init__(self, data, do_grad=False, dtype=None, device=None, fn=None):
        if device is None:
            device = svet.device.from_data(data) # imply device from data
        elif isinstance(device, str):
            device = svet.Device(device)
        # make sure device and data match (e.g. device="cpu" and
        # cupy.ndarray data do not match)
        assert device == svet.device.from_data(data)
        self.device = device
        self.backend, self.ops = device.backend, device.ops
        self.array = self.backend.asarray(data, dtype=dtype)

        self.do_grad = do_grad
        self._grad = None
        self.fn = fn

    def __neg__(self):
        return ag.NegFn()(self)

    def __mul__(self, other):
        return ag.MulFn()(self, other)

    def __rmul__(self, other):
        return ag.MulFn()(self, other)

    def __truediv__(self, other):
        return ag.DivFn()(self, other)

    def __floordiv__(self, other):
        return Tensor(self.array // strip(other))

    def __rtruediv__(self, other):
        return ag.RDivFn()(other, self)

    def __add__(self, other):
        return ag.AddFn()(self, other)

    def __radd__(self, other):
        return ag.AddFn()(self, other)

    def __sub__(self, other):
        return ag.SubFn()(self, other)

    def __rsub__(self, other):
        return ag.RSubFn()(other, self)

    def __isub__(self, other):
        if self.do_grad and ag.do_grad:
            raise TypeError("in-place operation is prohibited, since it may change the graph")
        # subtract directly, no need for SubFn here, since this op is
        # only allowed when gradient calculation is off
        self.array -= strip(other)
        return self

    def __imul__(self, other):
        if self.do_grad and ag.do_grad:
            raise TypeError("in-place operation is prohibited, since it may change the graph")
        # multiply directly, since gradient calculation is off
        self.array *= strip(other)
        return self

    def __abs__(self):
        return Tensor(abs(self.array))

    def __pow__(self, other):
        return ag.PowFn()(self, other)

    def __rpow__(self, other):
        return ag.RPowFn()(other, self)

    def __matmul__(self, other):
        return ag.MatMulFn()(self, other)

    def __rmatmul__(self, other):
        return ag.MatMulFn()(other, self)


    def __eq__(self, other):
        return Tensor(self.array == strip(other))

    def __ne__(self, other):
        return Tensor(self.array != strip(other))

    def __le__(self, other):
        return Tensor(self.array <= other.array)

    def __lt__(self, other):
        return Tensor(self.array < strip(other))

    def __gt__(self, other):
        return Tensor(self.array > strip(other))

    def __bool__(self):
        return bool(self.array)
    

    def __repr__(self):
        r = repr(self.array).replace("array", "tensor")
        if self.device.type == "cuda":
            r = r[:-1] + f", device='{self.device}')"
        if self.fn:
            r = r[:-1] + f", fn=<{self.fn.__class__.__name__}>)"
        elif self.do_grad:
            r = r[:-1] + ", do_grad=True)"
        return r
    
    def __getitem__(self, key):
        return ag.GetItemFn()(self, key)

    def __setitem__(self, key, val):
        self.array[key] = strip(val)

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        return iter(self.array)


    def cos(self):
        return ag.CosFn()(self)

    def sin(self):
        return ag.SinFn()(self)

    def sqrt(self):
        return ag.PowFn()(self, .5)

    def sum(self, **kws):
        return ag.SumFn()(self, **kws)

    def abs(self):
        return abs(self)


    def mean(self, axis=None, **kws):
        array = self.array
        if axis is None:
            n = array.size
        elif isinstance(axis, tuple):
            n = 1
            for a in axis:
                n *= array.shape[a]
        else:
            n = array.shape[axis]

        # It seems numpy converts n to float64 when dividing by it. In
        # cases where sum is reduced to a scalar this may promote the
        # (e.g. float32) result itself to float64. To keep original
        # dtype, convert n to it explicitly. Done only for float
        # types. This is the same behavior as numpy.mean.
        if numpy.issubdtype(array.dtype, numpy.floating):
            n = array.dtype.type(n)

        return self.sum(axis=axis, **kws) / n

    def var(self):
        return Tensor(self.array.var())

    def std(self):
        return Tensor(self.array.std())


    def exp(self):
        return ag.ExpFn()(self)

    def log(self):
        return ag.LogFn()(self)

    def log1p(self):
        return (self + self.array.dtype.type(1)).log()

    def sigmoid(self):
        return ag.SigmoidFn()(self)

    def reshape(self, shape):
        return ag.ReshapeFn()(self, shape)

    def argmax(self, **kws):
        return Tensor(self.array.argmax(**kws))

    def argsort(self, *args, **kws):
        return Tensor(self.array.argsort(*args, **kws))

    def max(self):
        return Tensor(self.array.max())

    def min(self):
        return Tensor(self.array.min())

    def all(self):
        return Tensor(self.array.all())


    def histc(self, bins=10, min=0, max=0):
        bounds = None
        if not min == max == 0:
            bounds = (min, max)
        return Tensor(self.backend.histogram(self.array, bins=bins, range=bounds)[0])

    def histogram(self, *args, **kws):
        hist, edges = self.backend.histogram(self.array, *args, **kws)
        return Tensor(hist), Tensor(edges)

    def zero_(self):
        if self.do_grad and ag.do_grad:
            raise TypeError("in-place operation is prohibited, since it may change the graph")
        # zero all elements, this works faster than creating new array
        # with zeros_like()
        self.array[...] = 0
        return self

    def to(self, dtype=None, device=None):
        old_dev, old_dt = self.device, self.array.dtype
        new_dev = old_dev if device is None else svet.device(device)
        new_dt = old_dt if dtype is None else dtype
        if new_dev == old_dev:
            if new_dt == old_dt:
                return self
            fn = ag.DTypeFn
        else:
            fn = ag.CPUFn if new_dev.type == "cpu" else ag.CUDAFn
        return fn()(self, dtype)

    def to_(self, dtype=None, device=None):
        with ag.no_grad():
            t = self.to(dtype=dtype, device=device)
            if t is self:
                return self
            self.array = t.array
            self.device = t.device
            self.backend = t.backend
            self.ops = t.ops
            if self._grad is not None:
                self._grad.to_(dtype=dtype, device=device)
            return self

    def float(self):
        return self.to(dtype=float32)

    def cuda(self):
        return self.to(device="cuda")

    def cpu(self):
        return self.to(device="cpu")


    def new_tensor(self, array, do_grad=False):
        return Tensor(array, dtype=self.array.dtype, device=self.device,
                      do_grad=do_grad)

    def backward(self, *args, **kws):
        if not self.do_grad:
            raise TypeError("this tensor doesn't require gradients")
        backward(self, *args, **kws)

    def do_grad_(self, do_grad=True):
        self.do_grad = do_grad
        return self

    # note, in torch this op is recorded in the graph, so grads coming
    # to cloned tensor also come to original one
    def clone(self):
        t = Tensor(self.array.copy(), do_grad=self.do_grad, fn=self.fn)
        if self._grad is not None:
            t._grad = Tensor(self._grad.array.copy())
        return t

    def detach_(self):
        self.do_grad = False
        self._grad = None
        self.fn = None
        return self

    def detach(self):
        return Tensor(self.array)

    def item(self):
        return self.array.item()


    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, other):
        self._grad = other

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def T(self):
        return Tensor(self.array.T)

    @property
    def ndim(self):
        return self.array.ndim


def tensor(data, device=None, **kws):
    if isinstance(data, (list, tuple)):
        data = [strip(el) for el in data]
    else:
        data = strip(data)
    return Tensor(data, device=device, **kws)


def empty(shape, dtype=None, do_grad=False, device=None):
    dev = svet.device.from_device(device)
    return Tensor(dev.backend.empty(shape, dtype=dtype),
                  do_grad=do_grad, device=dev)


def full(shape, fill_value, dtype=None, do_grad=False, device=None):
    dev = svet.device.from_device(device)
    return Tensor(dev.backend.full(shape, fill_value, dtype=dtype),
                  do_grad=do_grad, device=dev)


def zeros(shape, dtype=None, do_grad=False, device=None):
    dev = svet.device.from_device(device)
    return Tensor(dev.backend.zeros(shape, dtype=dtype),
                  do_grad=do_grad, device=dev)


def zeros_like(t, dtype=None, do_grad=False, device=None):
    if dtype is None:
        dtype = t.array.dtype
    if device is None:
        device = t.device
    return zeros(t.array.shape, dtype=dtype, do_grad=do_grad, device=device)


def ones(shape, dtype=None, do_grad=False, device=None):
    dev = svet.device.from_device(device)
    return Tensor(dev.backend.ones(shape, dtype=dtype),
                  do_grad=do_grad, device=dev)


def ones_like(t, dtype=None, do_grad=False, device=None):
    if dtype is None:
        dtype = t.array.dtype
    if device is None:
        device = t.device
    return ones(t.array.shape, dtype=dtype, do_grad=do_grad, device=device)


def arange(*args, dtype=None, do_grad=False, device=None):
    dev = svet.device.from_device(device)
    return Tensor(dev.backend.arange(*args, dtype=dtype),
                  do_grad=do_grad, device=dev)


def linspace(*args, dtype=None, do_grad=False, device=None, **kws):
    dev = svet.device.from_device(device)
    return Tensor(dev.backend.linspace(*args, **kws, dtype=dtype),
                  do_grad=do_grad, device=dev)


# We don't use global rng funcs from cupy, since it doesn't provide
# random.RandomState and we don't want to mess its global state from
# inside this library. So to get a random array on gpu it's first
# generated with our private numpy's RandomState and then moved to gpu
# inside Tensor constructor by means of the device arg. It must be
# slower than direct generation on gpu, but the upside is that we have
# same number sequences on cpu and gpu when seeded the same.

rndstate = numpy.random.RandomState()

def manual_seed(seed):
    rndstate.seed(seed)

def randn(*args, dtype=None, do_grad=False, device=None):
    return Tensor(rndstate.randn(*args), dtype=dtype,
                  do_grad=do_grad, device=device)

def randn_like(t, dtype=None, do_grad=False, device=None):
    if dtype is None:
        dtype = t.array.dtype
    if device is None:
        device = t.device
    return randn(*t.array.shape, dtype=dtype, do_grad=do_grad, device=device)

def rand(*args, dtype=None, do_grad=False, device=None):
    return Tensor(rndstate.rand(*args), dtype=dtype,
                  do_grad=do_grad, device=device)

def normal(mean, std, size, dtype=None, do_grad=False, device=None):
    return Tensor(rndstate.normal(mean, std, size), dtype=dtype,
                  do_grad=do_grad, device=device)

def randperm(n, dtype=None, do_grad=False, device=None):
    return Tensor(rndstate.permutation(n), dtype=dtype,
                  do_grad=do_grad, device=device)
