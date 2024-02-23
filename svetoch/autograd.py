import numpy

import svetoch.device
import svetoch.tensor as ten
from . import _cython_ops


do_grad = True
class no_grad:
    def __enter__(self):
        global do_grad
        do_grad = False

    def __exit__(self, *args):
        global do_grad
        do_grad = True

    def __call__(self, func):
        def wrapper_no_grad(*args, **kws):
            with self:
                return func(*args, **kws)
        return wrapper_no_grad


class Fn:
    def __call__(self, *args, **kws):
        # by default, args that are treated as tensors and thus may
        # need gradient, are positional args, but each func may change
        # that by setting self.args in its forward (e.g. may add a
        # tensor arg from its keyword args)
        self.args = args
        res = ten.Tensor(self.forward(*args, **kws))
        if do_grad:
            if len(self.args) == 1:
                res.requires_grad = self.args[0].requires_grad
            else:
                self.needs_grad = [
                    isinstance(a, ten.Tensor) and a.requires_grad
                    for a in self.args
                ]
                res.requires_grad = any(self.needs_grad)
            if res.requires_grad:
                res.fn = self
        return res

    def add_backward_hook(self, hook):
        if not hasattr(self, "bwd_hooks"):
            self.bwd_hooks = []
        self.bwd_hooks.append(hook)

    def get_args_grads(self, out_grad):
        grad = self.backward(out_grad)
        if hasattr(self, "bwd_hooks"):
            if isinstance(grad, tuple):
                tg = [None if g is None else ten.Tensor(g) for g in grad]
            else:
                tg = ten.Tensor(grad)
            for hook in self.bwd_hooks:
                hook(tg, ten.Tensor(out_grad))
        if len(self.args) == 1:
            return [(self.args[0], grad)]
        else:
            z = zip(self.args, self.needs_grad, grad)
            return [(a, g) for a, needs, g in z if needs]


@no_grad()
def backward(t, grad=None, create_graph=False):
    # since grad calculation is internal stuff, during graph traversal
    # running gradient is passed around as raw array instead of Tensor
    # object to avoid extra calls, but at leaves it is wrapped into Tensor
    if grad is None:
        grad = t.backend.ones_like(t.array)
    else:
        grad = t.backend.asarray(ten.strip(grad))
    assert t.array.shape == grad.shape, "shape of gradient doesn't match tensor"
    assert t.array.dtype == grad.dtype, "dtype of gradient doesn't match tensor"
    lst = [(t, grad)] # (tensor, running gradient)
    while lst:
        t, tgrad = lst.pop()
        # if tensor was broadcasted, so grad has different shape,
        # sum-reduce grad to original tensor shape
        array = t.array
        if tgrad.shape != array.shape:
            ddim = tgrad.ndim - array.ndim
            assert ddim >= 0, "broadcasting can't decrease num of dims"
            bcast = (1,) * ddim + array.shape if ddim > 0 else array.shape
            axes = tuple([
                i for i, (ng, nt) in enumerate(zip(tgrad.shape, bcast))
                if ng > nt
            ])
            if axes:
                # sum-reduce axes that were broadcasted
                tgrad = tgrad.sum(axis=axes, keepdims=True)
            # if broadcasting added axes, reshape to original
            if ddim > 0:
                tgrad = tgrad.reshape(array.shape)

        fn = t.fn
        if not fn or create_graph: # if leaf or saving grad to every node
            if t._grad is None:
                t._grad = ten.Tensor(tgrad)
            else:
                t._grad.array += tgrad

        if fn:
            lst.extend(fn.get_args_grads(tgrad))


class NegFn(Fn):
    def forward(self, x):
        return -x.array

    def backward(self, grad):
        return -grad


class AddFn(Fn):
    def forward(self, x, y):
        return x.array + ten.strip(y)

    def backward(self, grad):
        return grad, grad


class SubFn(Fn):
    def forward(self, x, y):
        return x.array - ten.strip(y)

    def backward(self, grad):
        return (
            grad,
            -grad if self.needs_grad[1] else None
    )


class RSubFn(SubFn):
    def forward(self, x, y):
        return x - y.array


class MulFn(Fn):
    def forward(self, x, y):
        x, y = x.array, ten.strip(y)
        self.saved = (x, y)
        return x * y

    def backward(self, grad):
        x, y = self.saved
        return (
            grad * y if self.needs_grad[0] else None,
            grad * x if self.needs_grad[1] else None
        )


class DivFn(Fn):
    def forward(self, x, y):
        x, y = x.array, ten.strip(y)
        self.saved = (x, y)
        return x / y

    def backward(self, grad):
        x, y = self.saved
        return (
            grad / y if self.needs_grad[0] else None,
            -grad * x / y**2. if self.needs_grad[1] else None
        )


class RDivFn(Fn):
    def forward(self, x, y):
        self.args = (y,) # only y may be a tensor
        y = y.array
        res = x / y
        self.saved = (res, y)
        return res

    def backward(self, grad):
        x_over_y, y = self.saved
        return -grad * x_over_y / y


class PowFn(Fn):
    def forward(self, x, y):
        backend, x, y = x.backend, x.array, ten.strip(y)
        self.saved = (backend, x, y)
        return x**y

    def backward(self, grad):
        backend, x, y = self.saved
        return (
            grad * y * x**(y - 1.) if self.needs_grad[0] else None,
            grad * x**y * backend.log(x) if self.needs_grad[1] else None
        )


class RPowFn(Fn):
    def forward(self, x, y):
        self.args = (y,) # only y may be a tensor
        res = x**y.array
        self.saved = (y.backend, x, res)
        return res

    def backward(self, grad):
        backend, x, x_pow_y = self.saved
        return grad * x_pow_y * backend.log(x)


class ExpFn(Fn):
    def forward(self, x):
        self.saved = x.backend.exp(x.array)
        return self.saved

    def backward(self, grad):
        return grad * self.saved


class LogFn(Fn):
    def forward(self, x):
        self.saved = x.array
        return x.backend.log(self.saved)

    def backward(self, grad):
        return grad / self.saved


class SigmoidFn(Fn):
    def forward(self, x):
        backend, x = x.backend, x.array
        # cast 1. to x.dtype to keep original type, otherwise numpy
        # will promote 1. (and the end result) to float64 when x is a
        # scalar
        one = x.dtype.type(1)
        res = one / (one + backend.exp(-x))
        self.saved = (one, res)
        return res

    def backward(self, grad):
        one, res = self.saved
        return grad * res * (one - res)


def generic_relu(dev, x):
    return dev.backend.maximum(x, 0.)


def generic_relu_backward(dev, x, y, y_grad):
    x_grad = y_grad.copy()
    # this op is slow and takes all of relu time, better alternatives?
    x_grad[y == 0.] = 0.
    return x_grad


class ReLUFn(Fn):
    def forward(self, x):
        dev, x = x.device, x.array
        y = dev.ops.relu(dev, x)
        self.saved = (dev, x, y)
        return y

    def backward(self, grad):
        dev, x, y = self.saved
        return dev.ops.relu_backward(dev, x, y, grad)


def generic_log_softmax(dev, x):
    # Plain softmax is unstable due to possible exp() result
    # overflow/underflow. Big positive x leads to inf result - overflow.
    # Big negative x leads to truncate-to-zero behaviour - underflow.
    # Due to softmax(x) == softmax(x + c) identity we can replace
    # softmax(x) with softmax(x - max(x)). z = x - max(x) leaves us with
    # negative values of z and at least one zero value (the maximum).
    # Now overflow can't happen, b/c z = 0 is the max value. Underflow
    # in the denominator also can't happen, b/c exp(0) gives 1.
    # Underflow in the numerator still can happen, though, and this is a
    # problem for log-softmax. To solve this we use the expanded form:
    # log(softmax(z)) = z - log(sum(exp(z))). In this form log's
    # argument is always at least 1.
    # See https://www.deeplearningbook.org/contents/numerical.html
    # Note, switch from x to z does not change differentiation, b/c
    # x_max is just a constant and dz = dx - d(x_max) = dx - 0 = dx.
    z = x - x.max(axis=1, keepdims=True)
    ez = dev.backend.exp(z)
    ezsum = ez.sum(axis=1, keepdims=True)
    y = z - dev.backend.log(ezsum)
    return y, (ez, ezsum)


def generic_log_softmax_backward(dev, y_grad, ez, ezsum):
    # Note that every forward output y_i value is a function not only of
    # its corresponding z_i, but of all z values due to the sum term in
    # softmax denominator. That means every y_i has gradient with respect
    # to every z value, not just z_i. Eventually those gradients sum up
    # so that every grad(z_i) includes gradients from all z values - this
    # is the y_grad.sum() term below.
    return y_grad - ez / ezsum * y_grad.sum(axis=1, keepdims=True)


class LogSoftmaxFn(Fn):
    def forward(self, x):
        dev, x = x.device, x.array
        y, saved = dev.ops.log_softmax(dev, x)
        self.saved = (dev, saved)
        return y

    def backward(self, grad):
        dev, args = self.saved
        return dev.ops.log_softmax_backward(dev, grad, *args)


class MatMulFn(Fn):
    def forward(self, x, y):
        x, y = x.array, y.array
        self.saved = (x, y)
        return x@y

    def backward(self, grad):
        x, y = self.saved
        return (
            grad@y.T if self.needs_grad[0] else None,
            x.T@grad if self.needs_grad[1] else None
        )


class GetItemFn(Fn):
    def forward(self, x, key):
        self.args = (x,)
        backend, x = x.backend, x.array
        if isinstance(key, tuple):
            key = tuple([ten.strip(k) for k in key])
        else:
            key = ten.strip(key)
        self.saved = (backend, x.shape, x.dtype, key)
        return x[key]

    def backward(self, grad):
        backend, shape, dtype, key = self.saved
        out = backend.zeros(shape, dtype=dtype)
        out[key] = grad
        return out


class ReshapeFn(Fn):
    def forward(self, x, shape):
        self.args = (x,)
        x = x.array
        self.saved = x.shape
        return x.reshape(shape)

    def backward(self, grad):
        return grad.reshape(self.saved)


class SumFn(Fn):
    def forward(self, x, axis=None, keepdims=False):
        xv = x.array
        self.saved = (xv, x.backend, axis, keepdims)
        # note: at least for numpy x.sum() is faster than np.sum(x)
        return xv.sum(axis=axis, keepdims=keepdims)

    def backward(self, grad):
        x, backend, axis, keepdims = self.saved
        # if axes were reduced, restore to broadcast grad correctly
        if not keepdims:
            if axis is None: axis = range(x.ndim)
            if isinstance(axis, int):
                grad = backend.expand_dims(grad, axis)
            else:
                # unlike numpy, cupy (as of 7.8) doesn't allow axis as
                # tuple in expand_dims, so we expand one dim at a time
                for ax in axis:
                    grad = backend.expand_dims(grad, ax)
        return backend.broadcast_to(grad, x.shape)


class CosFn(Fn):
    def forward(self, x):
        backend, x = x.backend, x.array
        self.saved = (backend, x)
        return backend.cos(x)

    def backward(self, grad):
        backend, x = self.saved
        return -grad * backend.sin(x)


class SinFn(Fn):
    def forward(self, x):
        backend, x = x.backend, x.array
        self.saved = (backend, x)
        return backend.sin(x)

    def backward(self, grad):
        backend, x = self.saved
        return grad * backend.cos(x)


# Here we bundle x @ weight + bias into a single op. This saves a graph
# node and some calculations. Backward formulas are taken from MatMulFn
# and AddFn. Also knowing that bias is broadcasted in a certain way
# avoids sum-reducing that would otherwise be done in a general way in
# backward(). Using this custom op gave 5% reduction in time.
class LinearFn(Fn):
    def forward(self, x, weight, bias):
        x, weight = x.array, weight.array
        # x (n, n_in) @ weight (n_in, n_out) = y (n, n_out)
        y = x @ weight
        if bias is not None:
            # broadcast bias (1, n_out) -> (n, n_out), then
            # y (n, n_out) += bias (n, n_out)
            y += bias.array
        self.saved = (x, weight)
        return y

    def backward(self, grad):
        x, weight = self.saved
        return (
            grad @ weight.T if self.needs_grad[0] else None,
            x.T @ grad if self.needs_grad[1] else None,
            grad.sum(axis=0, keepdims=True) if self.needs_grad[2] else None
        )


def generic_conv2d(dev, x, weight, stride, padding, h_out, w_out):
    n = x.shape[0]
    ch_out, ch_in, ksize_h, ksize_w = weight.shape
    c = ch_in * ksize_h * ksize_w
    xcol = dev.backend.empty((n, c, h_out * w_out), dtype=x.dtype)
    # for each output pixel extract input pixels used in its kernel
    # application into a separate column:
    # x (n, ch_in, h_in, w_in) -> xcol (n, c, h_out * w_out)
    dev.ops.im2col(x, ksize_h, ksize_w, stride, padding, h_out, w_out, xcol)
    # reshape, then broadcast weight (ch_out, c) -> (n, ch_out, c), then
    # weight (n, ch_out, c) @ xcol (n, c, h_out * w_out) =
    # y (n, ch_out, h_out * w_out)
    y = weight.reshape((ch_out, c)) @ xcol
    return y.reshape((n, ch_out, h_out, w_out)), xcol


def generic_conv2d_backward_x(dev, weight, y_grad, stride, padding,
        x_grad):
    ch_out, ch_in, ksize_h, ksize_w = weight.shape
    c = ch_in * ksize_h * ksize_w
    n, ch_out, h_out, w_out = y_grad.shape
    weight = weight.reshape((ch_out, c))
    # broadcast weight.T (c, ch_out) -> (n, c, ch_out), then
    # weight.T (n, c, ch_out) @ y_grad (n, ch_out, h_out * w_out) =
    # xcol_grad (n, c, h_out * w_out)
    xcol_grad = weight.T @ y_grad.reshape((n, ch_out, h_out * w_out))
    # accumulate gradients from extracted columns of kernel application
    # pixels back into input image pixels:
    # xcol_grad (n, c, h_out * w_out) -> x_grad (n, ch_in, h_in, w_in)
    dev.ops.col2im(xcol_grad, ksize_h, ksize_w, stride, padding,
                   h_out, w_out, x_grad)


def generic_conv2d_backward_weight(dev, xcol, y_grad, stride, padding,
        weight_grad):
    n, ch_out, h_out, w_out = y_grad.shape
    y_grad = y_grad.reshape((n, ch_out, h_out * w_out))
    # (ch_out, ch_in, ksize_h, ksize_w) -> (ch_out, c)
    weight_grad = weight_grad.reshape((ch_out, xcol.shape[1]))
    # transpose xcol (n, c, h_out * w_out) -> (n, h_out * w_out, c), then
    # y_grad (n, ch_out, h_out * w_out) @ xcol (n, h_out * w_out, c) =
    # weight_grad (n, ch_out, c), then sum weight gradients from all
    # images: sum weight_grad (n, ch_out, c) -> (ch_out, c)
    (y_grad @ xcol.transpose(0, 2, 1)).sum(axis=0, out=weight_grad)


class Conv2dFn(Fn):
    def forward(self, x, weight, bias, stride=1, padding=0):
        dev = x.device
        assert weight.device == dev
        x, weight = x.array, weight.array
        ch_out, ch_in, ksize_h, ksize_w = weight.shape
        n, ch_in_x, h_in, w_in = x.shape
        assert ch_in_x == ch_in

        #    3        6
        #  /-^-\ /----^----\
        #  K K K P P P P P P
        #  K K K . . . . . P
        #  K K K . . . . . P
        #  P . . . . . . . P
        #  P . . . . . . . P
        #  P . . . . . . . P
        #  P . . . . . . . P
        #  P . . . . . . . P
        #  P P P P P P P P P
        #  | \-----v-----/ |
        #  1   +   7   +   1   =   9
        #
        # The horizontal case:
        # - Input width (w_in) of 7 with a padding of 1 added to both
        #   left and right sides gives a total input width of 9:
        #
        #   w_in + 2 * padding = 7 + 2 * 1 = 9
        #
        # - The first output pixel is produced by applying kernel of
        #   size 3 (ksize_w) to the left edge (shown as K in image).
        #
        # - The rest of output pixels are produced by repeatedly moving
        #   kernel to the right and applying. The amount to move - row
        #   space that's left after applying the very first kernel:
        #
        #   w_in + 2 * padding - ksize_w = 9 - 3 = 6
        #
        # - 6 1-pixel moves to the right can be done and 6 more kernels
        #   applied giving 6 more output pixels (7 output pixels total).
        #
        # - If moving right 2 pixels (stride=2) at a time instead of 1,
        #   number of moves (and thus extra kernels applied) is halved:
        #
        #   stride=2: (w_in + 2 * padding - ksize_w) // stride = 6 // 2 = 3
        #   stride=3: 6 // 3 = 2
        #   stride=4: 6 // 4 = 1
        #
        # - To get the final number of output pixels, add the first one
        #   output pixel (produced by applying the kernel to left edge)
        #   to the number of output pixels produced by moving the kernel
        #   (i.e. to the number of moves):
        #
        #   (w_in + 2 * padding - ksize_w) // stride + 1 = 6 + 1 = 7
        #   stride=2: 6 // 2 + 1 = 4
        #   stride=3: 6 // 3 + 1 = 3
        #   stride=4: 6 // 4 + 1 = 2

        h_out = (h_in + 2 * padding - ksize_h) // stride + 1
        w_out = (w_in + 2 * padding - ksize_w) // stride + 1

        # Here we don't allocate y in advance and pass it to
        # ops.conv2d like in other ops.conv2d_* funcs. It's b/c
        # cupy.matmul doesn't yet support `out` param. Also
        # cudnn_conv2d returns original x as x_out for later backward,
        # whereas generic_conv2d returns unfolded im2col'ed x.
        y, x_out = dev.ops.conv2d(dev, x, weight, stride, padding, h_out, w_out)

        if bias is not None:
            assert bias.device == dev
            bias = bias.array
            assert bias.shape == (ch_out,)
            bias = bias.reshape((1, ch_out, 1, 1))
            # broadcast bias (1, ch_out, 1, 1) -> (n, ch_out, h_out, w_out)
            y += bias

        self.saved = (dev, x_out, x.shape, weight, stride, padding)
        return y

    def backward(self, grad):
        dev, x_out, x_shape, weight, stride, padding = self.saved
        if self.needs_grad[0]:
            x_grad = dev.backend.zeros(x_shape, dtype=x_out.dtype)
            dev.ops.conv2d_backward_x(dev, weight, grad, stride, padding,
                x_grad)

        if self.needs_grad[1]:
            weight_grad = dev.backend.empty_like(weight)
            dev.ops.conv2d_backward_weight(dev, x_out, grad, stride,
                padding, weight_grad)

        return (
            x_grad if self.needs_grad[0] else None,
            weight_grad if self.needs_grad[1] else None,
            # (n, ch_out, h_out, w_out) -> (ch_out,)
            grad.sum(axis=(0, 2, 3)) if self.needs_grad[2] else None
        )


class MaxPool2dFn(Fn):
    def forward(self, x, ksize, stride=1, padding=0):
        self.args = (x,)
        x = x.array
        out, idxs = _cython_ops.maxpool2d(x, ksize, stride, padding)
        self.saved = (idxs, *x.shape[2:])
        return out

    def backward(self, grad):
        idxs, h_in, w_in = self.saved
        return _cython_ops.maxpool2d_backward(grad, idxs, h_in, w_in)


class DTypeFn(Fn):
    def forward(self, x, dtype):
        self.args = (x,)
        x = x.array
        self.saved = x.dtype
        return x.astype(dtype)

    def backward(self, grad):
        old_dtype = self.saved
        return grad.astype(old_dtype)


class CuPyFn(Fn):
    def __init__(self):
        from svetoch.cuda import cupy
        if cupy is None:
            raise RuntimeError("cupy module is not available")
        self.cupy = cupy


class CUDAFn(CuPyFn):
    def forward(self, x, dtype=None):
        self.args = (x,)
        x = x.array
        self.saved = None if dtype is None else x.dtype
        return self.cupy.asarray(x, dtype=dtype)

    def backward(self, grad):
        old_dtype = self.saved
        res = self.cupy.asnumpy(grad)
        if old_dtype is None:
            return res
        return res.astype(old_dtype)


class CPUFn(CuPyFn):
    def forward(self, x, dtype=None):
        self.args = (x,)
        x = x.array
        res = self.cupy.asnumpy(x)
        if dtype is None:
            self.saved = None
            return res
        else:
            self.saved = x.dtype
            return res.astype(dtype)

    def backward(self, grad):
        old_dtype = self.saved
        return self.cupy.asarray(grad, dtype=old_dtype)


svetoch.device.register_device("cpu", numpy, dict(
    conv2d=generic_conv2d,
    conv2d_backward_x=generic_conv2d_backward_x,
    conv2d_backward_weight=generic_conv2d_backward_weight,
    im2col=_cython_ops.im2col,
    col2im=_cython_ops.col2im,
    relu=generic_relu,
    relu_backward=generic_relu_backward,
    log_softmax=generic_log_softmax,
    log_softmax_backward=generic_log_softmax_backward,
))
