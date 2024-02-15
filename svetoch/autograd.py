from collections import namedtuple

import numpy

from svetoch.tensor import Tensor, no_grad, strip
from . import _svetoch


try:
    import cupy
    from cupy import cudnn
    from cupy.cuda import cudnn as libcudnn
    cudnn_enabled = True # set to False to explicitly disable cudnn (used by tests)
except ImportError:
    cupy = None
    cudnn = None
    cudnn_enabled = False


def cuda_is_available():
    try:
        cupy.cuda.runtime.getDeviceCount()
    except:
        return False
    return True


class Fn:
    def __call__(self, *args, **kws):
        # by default, args that are treated as tensors and thus may
        # need gradient, are positional args, but each func may change
        # that by setting self.args in its forward (e.g. may add a
        # tensor arg from its keyword args)
        self.args = args
        res = Tensor(self.forward(*args, **kws))
        if Tensor.do_grad:
            if len(self.args) == 1:
                res.do_grad = self.args[0].do_grad
            else:
                self.needs_grad = [isinstance(a, Tensor) and a.do_grad
                                   for a in self.args]
                res.do_grad = any(self.needs_grad)
            if res.do_grad:
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
                tg = [None if g is None else Tensor(g) for g in grad]
            else:
                tg = Tensor(grad)
            for hook in self.bwd_hooks:
                hook(tg, Tensor(out_grad))
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
        grad = t.backend.asarray(strip(grad))
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
                t._grad = Tensor(tgrad)
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
        return x.array + strip(y)

    def backward(self, grad):
        return grad, grad


class SubFn(Fn):
    def forward(self, x, y):
        return x.array - strip(y)

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
        x, y = x.array, strip(y)
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
        x, y = x.array, strip(y)
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
        backend, x, y = x.backend, x.array, strip(y)
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


def cudnn_relu(dev, x):
    return cudnn.activation_forward(x, libcudnn.CUDNN_ACTIVATION_RELU)


def generic_relu(dev, x):
    return dev.backend.maximum(x, 0.)


def cudnn_relu_bwd(dev, x, y, y_grad):
    return cudnn.activation_backward(x, y, y_grad,
                                     libcudnn.CUDNN_ACTIVATION_RELU)


def generic_relu_bwd(dev, x, y, y_grad):
    x_grad = y_grad.copy()
    # this op is slow and takes all of relu time, better alternatives?
    x_grad[y == 0.] = 0.
    return x_grad


class ReLUFn(Fn):
    def forward(self, x):
        dev, xv = x.device, x.array
        y = dev.ops.relu(dev, xv)
        self.saved = (dev, xv, y)
        return y

    def backward(self, grad):
        dev, x, y = self.saved
        return dev.ops.relu_bwd(dev, x, y, grad)


def cudnn_log_softmax(dev, x):
    y = cudnn.softmax_forward(x, axis=1,
                              algorithm=libcudnn.CUDNN_SOFTMAX_LOG)
    return y, (y,)


def generic_log_softmax(dev, x):
    # Plain softmax is unstable due to possible exp()
    # overflow/underflow. Due to softmax(x) == softmax(x + c) identity
    # we can replace softmax(x) w/ softmax(x - max(x)). z = x - max(x)
    # leaves us negative values of z and one zero value which solves
    # instabilities for softmax. For log-softmax the problem of
    # softmax(z) = 0 still remains, so we use expanded form
    # log(softmax(z)) = z - log(sum(exp(z))), which solves that.
    z = x - x.max(axis=1, keepdims=True)
    ez = dev.backend.exp(z)
    ezsum = ez.sum(axis=1, keepdims=True)
    y = z - dev.backend.log(ezsum)
    return y, (ez, ezsum)


def cudnn_log_softmax_bwd(dev, y_grad, y):
    return cudnn.softmax_backward(y, y_grad, axis=1,
                                  algorithm=libcudnn.CUDNN_SOFTMAX_LOG)


def generic_log_softmax_bwd(dev, y_grad, ez, ezsum):
    return y_grad - ez / ezsum * y_grad.sum(axis=1, keepdims=True)


class LogSoftmaxFn(Fn):
    def forward(self, x):
        dev, x = x.device, x.array
        y, saved = dev.ops.log_softmax(dev, x)
        self.saved = (dev, saved)
        return y

    def backward(self, grad):
        dev, args = self.saved
        return dev.ops.log_softmax_bwd(dev, grad, *args)


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
            key = tuple([strip(k) for k in key])
        else:
            key = strip(key)
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


# Here we bundle x @ w + b into a single op. This saves a graph node and
# some calculations. Backward formulas are taken from MatMulFn and
# AddFn. Also knowing that bias would be broadcasted in a certain way
# avoids reducing that would otherwise be done in a general way in
# Tensor.backward(). Using this custom op instead of general code gave
# 5% reduction in time.
class LinearFn(Fn):
    def forward(self, x, w, b):
        x, w = x.array, w.array
        z = x @ w
        if b is not None: z += b.array
        self.saved = (x, w)
        return z

    def backward(self, grad):
        x, w = self.saved
        return (
            grad @ w.T if self.needs_grad[0] else None,
            x.T @ grad if self.needs_grad[1] else None,
            grad.sum(axis=0, keepdims=True) if self.needs_grad[2] else None
        )


cuda_cache={}

def cuda_im2col(x, ksize_h, ksize_w, stride, padding, h_out, w_out, out):
    assert x.flags.c_contiguous
    fn = 'im2col_' + x.dtype.name
    f = cuda_cache.get(fn, None)
    if f is None:
        raw = cupy.RawModule(code=r'''
template<typename T>
__device__ void im2col(
        const T *x, int N, int h_in, int w_in,
        int ksize_h, int ksize_w, int stride, int padding,
        int h_out, int w_out, T *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int sz = h_out * w_out;
    int c_in = idx / sz;
    idx = idx % sz;
    int i0 = (idx / w_out) * stride - padding;
    int j0 = (idx % w_out) * stride - padding;
    const T *x0 = x + (c_in * h_in + i0) * w_in + j0;
    T *pout = out + c_in * ksize_h * ksize_w * sz + idx;
    for (int h_off = 0; h_off < ksize_h; h_off++) {
        int i_in = i0 + h_off;
        for (int w_off = 0; w_off < ksize_w; w_off++) {
            int j_in = j0 + w_off;
            if (i_in < 0 || i_in >= h_in || j_in < 0 || j_in >= w_in) {
                *pout = 0.;
            } else {
                *pout = x0[h_off * w_in + w_off];
            }
            pout += sz;
        }
    }
}

// cupy only handles templated kernels starting from ver 8, so here we
// have to define separate wrappers for each float type
extern "C" {
__global__ void im2col_float32(
        const float *x, int N, int h_in, int w_in,
        int ksize_h, int ksize_w, int stride, int padding,
        int h_out, int w_out, float *out) {
    im2col<float>(x, N, h_in, w_in, ksize_h, ksize_w,
        stride, padding, h_out, w_out, out);
}

__global__ void im2col_float64(
        const double *x, int N, int h_in, int w_in,
        int ksize_h, int ksize_w, int stride, int padding,
        int h_out, int w_out, double *out) {
    im2col<double>(x, N, h_in, w_in, ksize_h, ksize_w,
        stride, padding, h_out, w_out, out);
}
}
''')
        f = raw.get_function(fn)
        cuda_cache[fn] = f

    n, ch_in, h_in, w_in = x.shape
    N = n * ch_in * h_out * w_out
    blk = 512
    grid = (N + blk - 1) // blk
    f((grid,),(blk,),(x, N, h_in, w_in, ksize_h, ksize_w,
                      stride, padding, h_out, w_out, out))


def cuda_col2im(grad, ksize_h, ksize_w, stride, padding, h_out, w_out, out):
    assert grad.flags.c_contiguous
    fn = 'col2im_' + grad.dtype.name
    f = cuda_cache.get(fn, None)
    if f is None:
        raw = cupy.RawModule(code=r'''
template<typename T>
__device__ void col2im(
        const T *grad, int N, int h_in, int w_in,
        int ksize_h, int ksize_w, int stride, int padding,
        int h_out, int w_out, T *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int sz = h_out * w_out;
    int c_in = idx / (h_in * w_in);
    int i_in = idx / w_in % h_in + padding;
    int j_in = idx % w_in + padding;

    // Find output pixels which use this input pixel.
    // (j_in - ksize_w) / stride + 1 below is the shortened version of
    // (j_in - (ksize_w - 1) + (stride - 1)) / stride, i.e. to find the leftmost
    // output pixel that uses this input pixel we first move left
    // across the width of the kernel, then by " + (stride - 1)) / stride"
    // (aka ceil() for ints) find the beginning of the next closest
    // output pixel from there. Note: due to int division these
    // versions are not exactly the same. For unused input pixels
    // (when stride > kernel width) the leftmost output pixel index will
    // be greater then the rightmost. Same logic applies for rows.
    int i_beg = i_in < ksize_h ? 0 : (i_in - ksize_h) / stride + 1;
    int j_beg = j_in < ksize_w ? 0 : (j_in - ksize_w) / stride + 1;
    int i_end = min(i_in / stride + 1, h_out);
    int j_end = min(j_in / stride + 1, w_out);

    // Algo being used in kernel is the optimized/incomprehensible
    // version of the following:
    //
    // /* start of channel */
    // const T *pg = grad + c_in * ksize_h * ksize_w * h_out * w_out;
    // for (int i = i_beg; i < i_end; i++) {
    //     for (int j = j_beg; j < j_end; j++) {
    //         /* start of receptive field inside channel */
    //         start = i * w_out + j;
    //         /* input pixel's offset inside receptive field,
    //            in range [0, ksize_h * ksize_w) */
    //         k = (i_in - i * stride) * ksize_w + (j_in - j * stride);
    //         val += pg[start + k * h_out * w_out];
    //     }
    // }
    const T *pg = grad + ((c_in * ksize_h + i_in) * ksize_w + j_in) * sz;
    int i_mul = w_out - stride * ksize_w * sz;
    int j_mul = 1 - stride * sz;
    T val = 0.;
    for (int i = i_beg; i < i_end; i++) {
        for (int j = j_beg; j < j_end; j++) {
            val += pg[i * i_mul + j * j_mul];
        }
    }
    out[idx] = val;
}

// cupy only handles templated kernels starting from ver 8, so here we
// have to define separate wrappers for each float type
extern "C" {
__global__ void col2im_float32(
        const float *grad, int N, int h_in, int w_in,
        int ksize_h, int ksize_w, int stride, int padding,
        int h_out, int w_out, float *out) {
    col2im<float>(grad, N, h_in, w_in,
        ksize_h, ksize_w, stride, padding, h_out, w_out, out);
}

__global__ void col2im_float64(
        const double *grad, int N, int h_in, int w_in,
        int ksize_h, int ksize_w, int stride, int padding,
        int h_out, int w_out, double *out) {
    col2im<double>(grad, N, h_in, w_in,
        ksize_h, ksize_w, stride, padding, h_out, w_out, out);
}
}
''')
        f = raw.get_function(fn)
        cuda_cache[fn] = f

    n, ch_in, h_in, w_in = out.shape
    N = n * ch_in * h_in * w_in
    blk = 512
    grid = (N + blk - 1) // blk
    f((grid,),(blk,),(grad, N, h_in, w_in, ksize_h, ksize_w,
                      stride, padding, h_out, w_out, out))


def cudnn_conv2d(dev, x, w, stride, padding, h_out, w_out):
    n, ch_out = x.shape[0], w.shape[0]
    y = dev.backend.empty((n, ch_out, h_out, w_out), dtype=x.dtype)
    cudnn.convolution_forward(x, w, None, y, (padding, padding),
                              (stride, stride), (1, 1), 1,
                              auto_tune=True, tensor_core='auto')
    return y, x


def generic_conv2d(dev, x, w, stride, padding, h_out, w_out):
    n = x.shape[0]
    ch_out, ch_in, ksize_h, ksize_w = w.shape
    c = ch_in * ksize_h * ksize_w
    xcol = dev.backend.empty((n, c, h_out * w_out), dtype=x.dtype)
    # (n, ch_in, h_in, w_in) -> (n, c, h_out * w_out)
    dev.ops.im2col(x, ksize_h, ksize_w, stride, padding, h_out, w_out, xcol)
    # bcast w (ch_out, c) -> (n, ch_out, c),
    # (n, ch_out, c) @ (n, c, h_out * w_out) = (n, ch_out, h_out * w_out)
    y = w.reshape((ch_out, c)) @ xcol
    return y.reshape((n, ch_out, h_out, w_out)), xcol


def cudnn_conv2d_bwd_x(dev, w, y_grad, stride, padding, x_grad):
    cudnn.convolution_backward_data(w, y_grad, None, x_grad,
                                    (padding, padding), (stride, stride),
                                    (1, 1), 1, deterministic=False,
                                     auto_tune=True, tensor_core='auto')


def generic_conv2d_bwd_x(dev, w, y_grad, stride, padding, x_grad):
    ch_out, ch_in, ksize_h, ksize_w = w.shape
    c = ch_in * ksize_h * ksize_w
    n, ch_out, h_out, w_out = y_grad.shape
    w = w.reshape((ch_out, c))
    # bcast w.T (c, ch_out) -> (n, c, ch_out),
    # (n, c, ch_out) @ (n, ch_out, h_out * w_out) = (n, c, h_out * w_out)
    xcol_grad = w.T @ y_grad.reshape((n, ch_out, h_out * w_out))
    # (n, c, h_out * w_out) -> (n, ch_in, h_in, w_in)
    dev.ops.col2im(xcol_grad, ksize_h, ksize_w, stride, padding,
                   h_out, w_out, x_grad)


def cudnn_conv2d_bwd_w(dev, x, y_grad, stride, padding, w_grad):
    cudnn.convolution_backward_filter(x, y_grad, w_grad, (padding, padding),
                                      (stride, stride), (1, 1), 1,
                                      deterministic=False,
                                      auto_tune=True, tensor_core='auto')


def generic_conv2d_bwd_w(dev, xcol, y_grad, stride, padding, w_grad):
    n, ch_out, h_out, w_out = y_grad.shape
    y_grad = y_grad.reshape((n, ch_out, h_out * w_out))
    # (ch_out, ch_in, ksize_h, ksize_w) -> (ch_out, c)
    w_grad = w_grad.reshape((ch_out, xcol.shape[1]))
    # transpose xcol (n, c, h_out * w_out) -> (n, h_out * w_out, c),
    # (n, ch_out, h_out * w_out) @ (n, h_out * w_out, c) = (n, ch_out, c),
    # sum (n, ch_out, c) -> (ch_out, c)
    (y_grad @ xcol.transpose(0, 2, 1)).sum(axis=0, out=w_grad)


class Conv2dFn(Fn):
    def forward(self, x, w, b, stride=1, padding=0):
        dev = x.device
        assert w.device == dev
        xv, w = x.array, w.array
        ch_out, ch_in, ksize_h, ksize_w = w.shape
        n, ch_in_x, h_in, w_in = xv.shape
        assert ch_in_x == ch_in

        h_out = (h_in + 2 * padding - ksize_h) // stride + 1
        w_out = (w_in + 2 * padding - ksize_w) // stride + 1
        # Here we don't allocate y in advance and pass it to
        # ops.conv2d like in other ops.conv2d_* funcs. It's b/c
        # cupy.matmul doesn't yet support `out` param. Also
        # cudnn_conv2d returns original x as x_out for later backward,
        # whereas generic_conv2d returns unfolded im2col'ed x.
        y, x_out = dev.ops.conv2d(dev, xv, w, stride, padding, h_out, w_out)

        if b is not None:
            assert b.device == dev
            b = b.array
            assert b.shape == (ch_out,)
            b = b.reshape((1, ch_out, 1, 1))
            # bcast b (1, ch_out, 1, 1) -> (n, ch_out, h_out, w_out)
            y += b

        self.saved = (dev, x_out, xv.shape, w, stride, padding)
        return y

    def backward(self, grad):
        dev, x_out, x_shape, w, stride, padding = self.saved
        if self.needs_grad[0]:
            x_grad = dev.backend.zeros(x_shape, dtype=x_out.dtype)
            dev.ops.conv2d_bwd_x(dev, w, grad, stride, padding, x_grad)

        if self.needs_grad[1]:
            w_grad = dev.backend.empty_like(w)
            dev.ops.conv2d_bwd_w(dev, x_out, grad, stride, padding, w_grad)

        return (
            x_grad if self.needs_grad[0] else None,
            w_grad if self.needs_grad[1] else None,
            # (n, ch_out, h_out, w_out) -> (ch_out, )
            grad.sum(axis=(0, 2, 3)) if self.needs_grad[2] else None
        )


class MaxPool2dFn(Fn):
    def forward(self, x, ksize, stride=1, padding=0):
        self.args = (x,)
        x = x.array
        out, idxs = _svetoch.maxpool2d(x, ksize, stride, padding)
        self.saved = (idxs, *x.shape[2:])
        return out

    def backward(self, grad):
        idxs, h_in, w_in = self.saved
        return _svetoch.maxpool2d_backward(grad, idxs, h_in, w_in)


class DTypeFn(Fn):
    def forward(self, x, dtype):
        self.args = (x,)
        x = x.array
        self.saved = x.dtype
        return x.astype(dtype)

    def backward(self, grad):
        old_dtype = self.saved
        return grad.astype(old_dtype)


class CUDAFn(Fn):
    def forward(self, x, dtype=None):
        self.args = (x,)
        x = x.array
        self.saved = None if dtype is None else x.dtype
        return cupy.asarray(x, dtype=dtype)

    def backward(self, grad):
        old_dtype = self.saved
        res = cupy.asnumpy(grad)
        if old_dtype is None: return res
        return res.astype(old_dtype)


class CPUFn(Fn):
    def forward(self, x, dtype=None):
        self.args = (x,)
        x = x.array
        res = cupy.asnumpy(x)
        if dtype is None:
            self.saved = None
            return res
        else:
            self.saved = x.dtype
            return res.astype(dtype)

    def backward(self, grad):
        old_dtype = self.saved
        return cupy.asarray(grad, dtype=old_dtype)


Ops = namedtuple('Ops', [
    'conv2d',
    'conv2d_bwd_x',
    'conv2d_bwd_w',
    'im2col',
    'col2im',
    'relu',
    'relu_bwd',
    'log_softmax',
    'log_softmax_bwd'
])

cpu_ops = Ops(
    generic_conv2d,
    generic_conv2d_bwd_x,
    generic_conv2d_bwd_w,
    _svetoch.im2col,
    _svetoch.col2im,
    generic_relu,
    generic_relu_bwd,
    generic_log_softmax,
    generic_log_softmax_bwd
)

cuda_ops = Ops(
    generic_conv2d,
    generic_conv2d_bwd_x,
    generic_conv2d_bwd_w,
    cuda_im2col,
    cuda_col2im,
    generic_relu,
    generic_relu_bwd,
    generic_log_softmax,
    generic_log_softmax_bwd
)

cudnn_ops = Ops(
    cudnn_conv2d,
    cudnn_conv2d_bwd_x,
    cudnn_conv2d_bwd_w,
    None,
    None,
    cudnn_relu,
    cudnn_relu_bwd,
    cudnn_log_softmax,
    cudnn_log_softmax_bwd
)


class Device:
    def __init__(self, type):
        self.type = type
        if type == "cpu":
            self.backend, self.ops = numpy, cpu_ops
        elif type == "cuda":
            if cupy is None:
                raise ModuleNotFoundError("cupy module is not available")
            self.backend = cupy
            if cudnn is not None and cudnn_enabled:
                self.ops = cudnn_ops
            else:
                self.ops = cuda_ops
        else:
            raise ValueError("device type must be 'cpu' or 'cuda'")

    def __repr__(self):
        return self.type

    __str__ = __repr__

    def __eq__(self, other):
        return self.type == other.type

    @staticmethod
    def from_data(data):
        if cupy is not None and isinstance(data, cupy.ndarray):
            return Device("cuda")
        return Device("cpu")

    @staticmethod
    def from_device(device=None):
        if device is None:
            return Device("cpu")
        elif isinstance(device, str):
            return Device(device)
        return device
