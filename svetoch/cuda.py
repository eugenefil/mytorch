import svetoch as svet


try:
    import cupy
    from cupy import cudnn
    from cupy.cuda import cudnn as libcudnn
    cudnn_enabled = True # set to False to explicitly disable cudnn (used by tests)
except ImportError:
    cupy = None
    cudnn = None
    cudnn_enabled = False


def is_available():
    try:
        cupy.cuda.runtime.getDeviceCount()
    except:
        return False
    return True


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


def cudnn_conv2d(dev, x, weight, stride, padding, h_out, w_out):
    n, ch_out = x.shape[0], weight.shape[0]
    y = dev.backend.empty((n, ch_out, h_out, w_out), dtype=x.dtype)
    cudnn.convolution_forward(x, weight, None, y, (padding, padding),
                              (stride, stride), (1, 1), 1,
                              auto_tune=True, tensor_core='auto')
    return y, x


def cudnn_conv2d_bwd_x(dev, weight, y_grad, stride, padding, x_grad):
    cudnn.convolution_backward_data(weight, y_grad, None, x_grad,
                                    (padding, padding), (stride, stride),
                                    (1, 1), 1, deterministic=False,
                                     auto_tune=True, tensor_core='auto')


def cudnn_conv2d_bwd_w(dev, x, y_grad, stride, padding, w_grad):
    cudnn.convolution_backward_filter(x, y_grad, w_grad, (padding, padding),
                                      (stride, stride), (1, 1), 1,
                                      deterministic=False,
                                      auto_tune=True, tensor_core='auto')


def cudnn_relu(dev, x):
    return cudnn.activation_forward(x, libcudnn.CUDNN_ACTIVATION_RELU)


def cudnn_relu_bwd(dev, x, y, y_grad):
    return cudnn.activation_backward(x, y, y_grad,
                                     libcudnn.CUDNN_ACTIVATION_RELU)


def cudnn_log_softmax(dev, x):
    y = cudnn.softmax_forward(x, axis=1,
                              algorithm=libcudnn.CUDNN_SOFTMAX_LOG)
    return y, (y,)


def cudnn_log_softmax_bwd(dev, y_grad, y):
    return cudnn.softmax_backward(y, y_grad, axis=1,
                                  algorithm=libcudnn.CUDNN_SOFTMAX_LOG)


def probe_data(data):
    return isinstance(data, cupy.ndarray)


if cupy is not None:
    if cudnn is not None and cudnn_enabled:
        svet.device.register_device('cuda', cupy, {
            "conv2d": cudnn_conv2d,
            "conv2d_bwd_x": cudnn_conv2d_bwd_x,
            "conv2d_bwd_w": cudnn_conv2d_bwd_w,
            "relu": cudnn_relu,
            "relu_bwd": cudnn_relu_bwd,
            "log_softmax": cudnn_log_softmax,
            "log_softmax_bwd": cudnn_log_softmax_bwd,
        }, probe_data=probe_data)
    else:
        svet.device.register_ops('cuda', cupy, {
            "im2col": cuda_im2col,
            "col2im": cuda_col2im,
        })
