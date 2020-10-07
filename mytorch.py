from collections import namedtuple

import numpy as np
import cupy as cp

import _mytorch

try:
    from cupy import cudnn
    from cupy.cuda import cudnn as libcudnn
    cudnn_enabled=True
except ImportError:
    cudnn=None
    cudnn_enabled=False

### AUTOGRAD ###

class no_grad:
    def __enter__(self):
        Tensor.do_grad=False
    def __exit__(self,*args):
        Tensor.do_grad=True
    def __call__(self,func):
        def wrapper_no_grad(*args,**kws):
            with self:
                return func(*args,**kws)
        return wrapper_no_grad


class Fn:
    def __call__(self,*args,**kws):
        # by default, args that are treated as tensors and thus may
        # need gradient, are positional args, but each func may change
        # that by setting self.args in its forward (e.g. may add a
        # tensor arg from its keyword args)
        self.args=args
        res=Tensor(self.forward(*args,**kws))
        if Tensor.do_grad:
            if len(self.args)==1:
                res.do_grad=self.args[0].do_grad
            else:
                self.needs_grad=[isinstance(a,Tensor) and a.do_grad
                                 for a in self.args]
                res.do_grad=any(self.needs_grad)
            if res.do_grad: res.fn=self
        return res

    def get_args_grads(self,grad_in):
        grad=self.backward(grad_in)
        if len(self.args)==1:
            return [(self.args[0],grad)]
        else:
            z=zip(self.args,self.needs_grad,grad)
            return [(a,g) for a,needs,g in z if needs]


@no_grad()
def backward(t,grad=None,create_graph=False):
    # since grad calculation is internal stuff, during graph traversal
    # running gradient is passed around as raw array instead of Tensor
    # object to avoid extra calls, but at leaves it is wrapped into
    # Tensor
    if grad is None: grad=t.am.ones_like(t.v)
    else: grad=t.am.asarray(strip(grad))
    assert t.v.shape==grad.shape,"shape of gradient doesn't match tensor"
    assert t.v.dtype==grad.dtype,"dtype of gradient doesn't match tensor"
    lst=[(t,grad)] # (tensor,running gradient)
    while lst:
        t,tgrad=lst.pop()
        # if tensor was broadcasted, so grad has different shape,
        # sum-reduce grad to original tensor shape
        v=t.v
        if tgrad.shape!=v.shape:
            ddim=tgrad.ndim-v.ndim
            assert ddim>=0, "broadcasting can't decrease num of dims"
            bcast=(1,)*ddim+v.shape if ddim>0 else v.shape
            axes=tuple([
                i for i,(ng,nt) in enumerate(zip(tgrad.shape,bcast))
                if ng>nt
            ])
            # sum-reduce axes that were broadcasted
            if axes: tgrad=tgrad.sum(axis=axes,keepdims=True)
            # if broadcasting added axes, reshape to original
            if ddim>0: tgrad=tgrad.reshape(v.shape)

        fn=t.fn
        if not fn or create_graph: # if leaf or saving grad to every node
            if t._grad is None:
                t._grad=Tensor(tgrad)
            else:
                t._grad.v+=tgrad

        if fn: lst.extend(fn.get_args_grads(tgrad))


class NegFn(Fn):
    def forward(self,x): return -x.v
    def backward(self,grad): return -grad

class AddFn(Fn):
    def forward(self,x,y): return x.v+strip(y)
    def backward(self,grad): return grad,grad

class SubFn(Fn):
    def forward(self,x,y): return x.v-strip(y)
    def backward(self,grad): return (
            grad,
            -grad if self.needs_grad[1] else None
    )

class RSubFn(SubFn):
    def forward(self,x,y): return x-y.v

class MulFn(Fn):
    def forward(self,x,y):
        x,y=x.v,strip(y)
        self.saved=(x,y)
        return x*y

    def backward(self,grad):
        x,y=self.saved
        return (
            grad*y if self.needs_grad[0] else None,
            grad*x if self.needs_grad[1] else None
        )

class DivFn(Fn):
    def forward(self,x,y):
        x,y=x.v,strip(y)
        self.saved=(x,y)
        return x/y

    def backward(self,grad):
        x,y=self.saved
        return (
            grad/y if self.needs_grad[0] else None,
            -grad*x/y**2. if self.needs_grad[1] else None
        )

class RDivFn(Fn):
    def forward(self,x,y):
        self.args=(y,) # only y may be a tensor
        y=y.v
        res=x/y
        self.saved=(res,y)
        return res

    def backward(self,grad):
        x_over_y,y=self.saved
        return -grad*x_over_y/y

class PowFn(Fn):
    def forward(self,x,y):
        am,x,y=x.am,x.v,strip(y)
        self.saved=(am,x,y)
        return x**y

    def backward(self,grad):
        am,x,y=self.saved
        return (
            grad*y*x**(y-1.) if self.needs_grad[0] else None,
            grad*x**y*am.log(x) if self.needs_grad[1] else None
        )

class RPowFn(Fn):
    def forward(self,x,y):
        self.args=(y,) # only y may be a tensor
        res=x**y.v
        self.saved=(y.am,x,res)
        return res

    def backward(self,grad):
        am,x,x_pow_y=self.saved
        return grad*x_pow_y*am.log(x)

class ExpFn(Fn):
    def forward(self,x):
        self.saved=x.am.exp(x.v)
        return self.saved

    def backward(self,grad): return grad*self.saved

class LogFn(Fn):
    def forward(self,x):
        self.saved=x.v
        return x.am.log(self.saved)

    def backward(self,grad): return grad/self.saved

class SigmoidFn(Fn):
    def forward(self,x):
        am,x=x.am,x.v
        # cast 1. to x.dtype to keep original type, otherwise numpy
        # will promote 1. (and the end result) to float64 when x is a
        # scalar
        one=x.dtype.type(1)
        res=one/(one+am.exp(-x))
        self.saved=(one,res)
        return res

    def backward(self,grad):
        one,res=self.saved
        return grad*res*(one-res)

def cudnn_relu(dev,x):
    return cudnn.activation_forward(x,libcudnn.CUDNN_ACTIVATION_RELU)

def generic_relu(dev,x): return dev.am.maximum(x,0.)

def cudnn_relu_bwd(dev,x,y,y_grad):
    return cudnn.activation_backward(x,y,y_grad,
                                     libcudnn.CUDNN_ACTIVATION_RELU)

def generic_relu_bwd(dev,x,y,y_grad):
    x_grad=y_grad.copy()
    # this op is slow and takes all of relu time, better alternatives?
    x_grad[y==0.]=0.
    return x_grad

class ReLUFn(Fn):
    def forward(self,x):
        dev,xv=x.device,x.v
        y=dev.aux.relu(dev,xv)
        self.saved=(dev,xv,y)
        return y

    def backward(self,grad):
        dev,x,y=self.saved
        return dev.aux.relu_bwd(dev,x,y,grad)

class LogSoftmaxFn(Fn):
    def forward(self,x):
        # Plain softmax is unstable due to possible exp()
        # overflow/underflow. Due to softmax(x) == softmax(x+c)
        # identity we can replace softmax(x) w/
        # softmax(x-max(x)). z=x-max(x) leaves us negative values of z
        # and one zero value which solves instabilities for
        # softmax. For log-softmax the problem of softmax(z)=0 still
        # remains, so we use expanded form log(softmax(z)) =
        # z-log(sum(exp(z))), which solves that.
        am,x=x.am,x.v
        z=x-x.max(axis=1,keepdims=True)
        ez=am.exp(z)
        ezsum=ez.sum(axis=1,keepdims=True)
        self.saved=(ez,ezsum)
        return z-am.log(ezsum)

    def backward(self,grad):
        ez,ezsum=self.saved
        return grad-ez/ezsum*grad.sum(axis=1,keepdims=True)

class MatMulFn(Fn):
    def forward(self,x,y):
        x,y=x.v,y.v
        self.saved=(x,y)
        return x@y

    def backward(self,grad):
        x,y=self.saved
        return (
            grad@y.T if self.needs_grad[0] else None,
            x.T@grad if self.needs_grad[1] else None
        )

class GetItemFn(Fn):
    def forward(self,x,key):
        self.args=(x,)
        am,x=x.am,x.v
        if isinstance(key,tuple):
            key=tuple([strip(k) for k in key])
        else:
            key=strip(key)
        self.saved=(am,x.shape,x.dtype,key)
        return x[key]

    def backward(self,grad):
        am,shape,dtype,key=self.saved
        out=am.zeros(shape,dtype=dtype)
        out[key]=grad
        return out

class ReshapeFn(Fn):
    def forward(self,x,shape):
        self.args=(x,)
        x=x.v
        self.saved=x.shape
        return x.reshape(shape)

    def backward(self,grad): return grad.reshape(self.saved)

class SumFn(Fn):
    def forward(self,x,axis=None,keepdims=False):
        xv=x.v
        self.saved=(xv,x.am,axis,keepdims)
        # note: at least for numpy x.sum() is faster than np.sum(x)
        return xv.sum(axis=axis,keepdims=keepdims)

    def backward(self,grad):
        x,am,axis,keepdims=self.saved
        # if axes were reduced, restore to broadcast grad correctly
        if not keepdims:
            if axis is None: axis=range(x.ndim)
            if isinstance(axis,int):
                grad=am.expand_dims(grad,axis)
            else:
                # unlike numpy, cupy (as of 7.8) doesn't allow axis as
                # tuple in expand_dims, so we expand one dim at a time
                for ax in axis:
                    grad=am.expand_dims(grad,ax)
        return am.broadcast_to(grad,x.shape)

class CosFn(Fn):
    def forward(self,x):
        am,x=x.am,x.v
        self.saved=(am,x)
        return am.cos(x)

    def backward(self,grad):
        am,x=self.saved
        return -grad*am.sin(x)

class SinFn(Fn):
    def forward(self,x):
        am,x=x.am,x.v
        self.saved=(am,x)
        return am.sin(x)

    def backward(self,grad):
        am,x=self.saved
        return grad*am.cos(x)

# Here we bundle x@w+b into a single op. This saves a graph node and
# some calculations. Backward formulas are taken from MatMulFn and
# AddFn. Also knowing that bias would be broadcasted in a certain way
# avoids reducing that would otherwise be done in a general way in
# Tensor.backward(). Using this custom op instead of general code gave
# 5% reduction in time.
class LinearFn(Fn):
    def forward(self,x,w,b):
        x,w=x.v,w.v
        z=x@w
        if b is not None: z+=b.v
        self.saved=(x,w)
        return z

    def backward(self,grad):
        x,w=self.saved
        return (
            grad@w.T if self.needs_grad[0] else None,
            x.T@grad if self.needs_grad[1] else None,
            grad.sum(axis=0,keepdims=True) if self.needs_grad[2] else None
        )

cuda_cache={}

def cuda_im2col(x,ksize_h,ksize_w,stride,padding,h_out,w_out,out):
    assert x.flags.c_contiguous
    fn='im2col_'+x.dtype.name
    f=cuda_cache.get(fn,None)
    if f is None:
        raw=cp.RawModule(code=r'''
template<typename T>
__device__ void im2col(
        const T *x,int N,int h_in,int w_in,
        int ksize_h,int ksize_w,int stride,int padding,
        int h_out,int w_out,T *out) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=N) return;
    int sz=h_out*w_out;
    int c_in=idx/sz;
    idx=idx%sz;
    int i0=(idx/w_out)*stride-padding;
    int j0=(idx%w_out)*stride-padding;
    const T *x0=x+(c_in*h_in+i0)*w_in+j0;
    T *pout=out+c_in*ksize_h*ksize_w*sz+idx;
    for (int h_off=0;h_off<ksize_h;h_off++) {
        int i_in=i0+h_off;
        for (int w_off=0;w_off<ksize_w;w_off++) {
            int j_in=j0+w_off;
            if (i_in<0 || i_in>=h_in || j_in<0 || j_in>=w_in) {
                *pout=0.;
            } else {
                *pout=x0[h_off*w_in+w_off];
            }
            pout+=sz;
        }
    }
}

// cupy only handles templated kernels starting from ver 8, so here we
// have to define separate wrappers for each float type
extern "C" {
__global__ void im2col_float32(
        const float *x,int N,int h_in,int w_in,
        int ksize_h,int ksize_w,int stride,int padding,
        int h_out,int w_out,float *out) {
    im2col<float>(x,N,h_in,w_in,ksize_h,ksize_w,
        stride,padding,h_out,w_out,out);
}

__global__ void im2col_float64(
        const double *x,int N,int h_in,int w_in,
        int ksize_h,int ksize_w,int stride,int padding,
        int h_out,int w_out,double *out) {
    im2col<double>(x,N,h_in,w_in,ksize_h,ksize_w,
        stride,padding,h_out,w_out,out);
}
}
''')
        f=raw.get_function(fn)
        cuda_cache[fn]=f

    n,ch_in,h_in,w_in=x.shape
    N=n*ch_in*h_out*w_out
    blk=512
    grid=(N+blk-1)//blk
    f((grid,),(blk,),(x,N,h_in,w_in,ksize_h,ksize_w,
                      stride,padding,h_out,w_out,out))

def cuda_col2im(grad,ksize_h,ksize_w,stride,padding,h_out,w_out,out):
    assert grad.flags.c_contiguous
    fn='col2im_'+grad.dtype.name
    f=cuda_cache.get(fn,None)
    if f is None:
        raw=cp.RawModule(code=r'''
template<typename T>
__device__ void col2im(
        const T *grad,int N,int h_in,int w_in,
        int ksize_h,int ksize_w,int stride,int padding,
        int h_out,int w_out,T *out) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=N) return;
    int sz=h_out*w_out;
    int c_in=idx/(h_in*w_in);
    int i_in=idx/w_in%h_in+padding;
    int j_in=idx%w_in+padding;

    // Find output pixels which use this input pixel.
    // (j_in-ksize_w)/stride+1 below is the shortened version of
    // (j_in-(ksize_w-1)+(stride-1))/stride, i.e. to find the leftmost
    // output pixel that uses this input pixel we first move left
    // across the width of the kernel, then by "+(stride-1))/stride"
    // (aka ceil() for ints) find the beginning of the next closest
    // output pixel from there. Note: due to int division these
    // versions are not exactly the same. For unused input pixels
    // (when stride>kernel width) the leftmost output pixel index will
    // be greater then the rightmost. Same logic applies for rows.
    int i_beg=i_in<ksize_h ? 0 : (i_in-ksize_h)/stride+1;
    int j_beg=j_in<ksize_w ? 0 : (j_in-ksize_w)/stride+1;
    int i_end=min(i_in/stride+1,h_out);
    int j_end=min(j_in/stride+1,w_out);

    // Algo being used in kernel is the optimized/incomprehensible
    // version of the following:
    //
    // /* start of channel */
    // const T *pg=grad+c_in*ksize_h*ksize_w*h_out*w_out;
    // for (int i=i_beg;i<i_end;i++) {
    //     for (int j=j_beg;j<j_end;j++) {
    //         /* start of receptive field inside channel */
    //         start=i*w_out+j;
    //         /* input pixel's offset inside receptive field,
    //            in range [0,ksize_h*ksize_w) */
    //         k=(i_in-i*stride)*ksize_w+(j_in-j*stride);
    //         val+=pg[start+k*h_out*w_out];
    //     }
    // }
    const T *pg=grad+((c_in*ksize_h+i_in)*ksize_w+j_in)*sz;
    int i_mul=w_out-stride*ksize_w*sz;
    int j_mul=1-stride*sz;
    T val=0.;
    for (int i=i_beg;i<i_end;i++) {
        for (int j=j_beg;j<j_end;j++) {
            val+=pg[i*i_mul+j*j_mul];
        }
    }
    out[idx]=val;
}

// cupy only handles templated kernels starting from ver 8, so here we
// have to define separate wrappers for each float type
extern "C" {
__global__ void col2im_float32(
        const float *grad,int N,int h_in,int w_in,
        int ksize_h,int ksize_w,int stride,int padding,
        int h_out,int w_out,float *out) {
    col2im<float>(grad,N,h_in,w_in,
        ksize_h,ksize_w,stride,padding,h_out,w_out,out);
}

__global__ void col2im_float64(
        const double *grad,int N,int h_in,int w_in,
        int ksize_h,int ksize_w,int stride,int padding,
        int h_out,int w_out,double *out) {
    col2im<double>(grad,N,h_in,w_in,
        ksize_h,ksize_w,stride,padding,h_out,w_out,out);
}
}
''')
        f=raw.get_function(fn)
        cuda_cache[fn]=f

    n,ch_in,h_in,w_in=out.shape
    N=n*ch_in*h_in*w_in
    blk=512
    grid=(N+blk-1)//blk
    f((grid,),(blk,),(grad,N,h_in,w_in,ksize_h,ksize_w,
                      stride,padding,h_out,w_out,out))

def cudnn_conv2d(dev,x,w,stride,padding,h_out,w_out):
    n,ch_out=x.shape[0],w.shape[0]
    y=dev.am.empty((n,ch_out,h_out,w_out),dtype=x.dtype)
    cudnn.convolution_forward(x,w,None,y,(padding,padding),
                              (stride,stride),(1,1),1,
                              auto_tune=True,tensor_core='auto')
    return y,x

def generic_conv2d(dev,x,w,stride,padding,h_out,w_out):
    n=x.shape[0]
    ch_out,ch_in,ksize_h,ksize_w=w.shape
    c=ch_in*ksize_h*ksize_w
    xcol=dev.am.empty((n,c,h_out*w_out),dtype=x.dtype)
    # (n,ch_in,h_in,w_in) -> (n,c,h_out*w_out)
    dev.aux.im2col(x,ksize_h,ksize_w,stride,padding,h_out,w_out,xcol)
    # bcast w (ch_out,c) -> (n,ch_out,c),
    # (n,ch_out,c) @ (n,c,h_out*w_out) = (n,ch_out,h_out*w_out)
    y=w.reshape((ch_out,c))@xcol
    return y.reshape((n,ch_out,h_out,w_out)),xcol

def cudnn_conv2d_bwd_x(dev,w,y_grad,stride,padding,x_grad):
    cudnn.convolution_backward_data(w,y_grad,None,x_grad,
                                    (padding,padding),(stride,stride),
                                    (1,1),1,deterministic=False,
                                     auto_tune=True,tensor_core='auto')

def generic_conv2d_bwd_x(dev,w,y_grad,stride,padding,x_grad):
    ch_out,ch_in,ksize_h,ksize_w=w.shape
    c=ch_in*ksize_h*ksize_w
    n,ch_out,h_out,w_out=y_grad.shape
    w=w.reshape((ch_out,c))
    # bcast w.T (c,ch_out) -> (n,c,ch_out),
    # (n,c,ch_out) @ (n,ch_out,h_out*w_out) = (n,c,h_out*w_out)
    xcol_grad=w.T@y_grad.reshape((n,ch_out,h_out*w_out))
    # (n,c,h_out*w_out) -> (n,ch_in,h_in,w_in)
    dev.aux.col2im(xcol_grad,ksize_h,ksize_w,stride,padding,
                   h_out,w_out,x_grad)

def cudnn_conv2d_bwd_w(dev,x,y_grad,stride,padding,w_grad):
    cudnn.convolution_backward_filter(x,y_grad,w_grad,(padding,padding),
                                      (stride,stride),(1,1),1,
                                      deterministic=False,
                                      auto_tune=True,tensor_core='auto')

def generic_conv2d_bwd_w(dev,xcol,y_grad,stride,padding,w_grad):
    n,ch_out,h_out,w_out=y_grad.shape
    y_grad=y_grad.reshape((n,ch_out,h_out*w_out))
    # (ch_out,ch_in,ksize_h,ksize_w) -> (ch_out,c)
    w_grad=w_grad.reshape((ch_out,xcol.shape[1]))
    # transpose xcol (n,c,h_out*w_out) -> (n,h_out*w_out,c),
    # (n,ch_out,h_out*w_out) @ (n,h_out*w_out,c) = (n,ch_out,c),
    # sum (n,ch_out,c) -> (ch_out,c)
    (y_grad@xcol.transpose(0,2,1)).sum(axis=0,out=w_grad)

class Conv2dFn(Fn):
    def forward(self,x,w,b,stride=1,padding=0):
        dev=x.device
        assert w.device==dev
        xv,w=x.v,w.v
        ch_out,ch_in,ksize_h,ksize_w=w.shape
        n,ch_in_x,h_in,w_in=xv.shape
        assert ch_in_x==ch_in

        h_out=(h_in+2*padding-ksize_h)//stride+1
        w_out=(w_in+2*padding-ksize_w)//stride+1
        # Here we don't allocate y in advance and pass it to
        # aux.conv2d like in other aux.conv2d_* funcs. It's b/c
        # cupy.matmul doesn't yet support `out` param. Also
        # cudnn_conv2d returns original x as x_out for later backward,
        # whereas generic_conv2d returns unfolded im2col'ed x.
        y,x_out=dev.aux.conv2d(dev,xv,w,stride,padding,h_out,w_out)

        if b is not None:
            assert b.device==dev
            b=b.v
            assert b.shape==(ch_out,)
            b=b.reshape((1,ch_out,1,1))
            # bcast b (1,ch_out,1,1) -> (n,ch_out,h_out,w_out)
            y+=b

        self.saved=(dev,x_out,xv.shape,w,stride,padding)
        return y

    def backward(self,grad):
        dev,x_out,x_shape,w,stride,padding=self.saved
        if self.needs_grad[0]:
            x_grad=dev.am.zeros(x_shape,dtype=x_out.dtype)
            dev.aux.conv2d_bwd_x(dev,w,grad,stride,padding,x_grad)

        if self.needs_grad[1]:
            w_grad=dev.am.empty_like(w)
            dev.aux.conv2d_bwd_w(dev,x_out,grad,stride,padding,w_grad)

        return (
            x_grad if self.needs_grad[0] else None,
            w_grad if self.needs_grad[1] else None,
            # (n,ch_out,h_out,w_out) -> (ch_out,)
            grad.sum(axis=(0,2,3)) if self.needs_grad[2] else None
        )

class MaxPool2dFn(Fn):
    def forward(self,x,ksize,stride=1,padding=0):
        self.args=(x,)
        x=x.v
        out,idxs=_mytorch.maxpool2d(x,ksize,stride,padding)
        self.saved=(idxs,*x.shape[2:])
        return out

    def backward(self,grad):
        idxs,h_in,w_in=self.saved
        return _mytorch.maxpool2d_backward(grad,idxs,h_in,w_in)

class DTypeFn(Fn):
    def forward(self,x,dtype):
        self.args=(x,)
        x=x.v
        self.saved=x.dtype
        return x.astype(dtype)

    def backward(self,grad):
        old_dtype=self.saved
        return grad.astype(old_dtype)

class CUDAFn(Fn):
    def forward(self,x,dtype=None):
        self.args=(x,)
        x=x.v
        self.saved=None if dtype is None else x.dtype
        return cp.asarray(x,dtype=dtype)

    def backward(self,grad):
        old_dtype=self.saved
        res=cp.asnumpy(grad)
        if old_dtype is None: return res
        return res.astype(old_dtype)

class CPUFn(Fn):
    def forward(self,x,dtype=None):
        self.args=(x,)
        x=x.v
        res=cp.asnumpy(x)
        if dtype is None:
            self.saved=None
            return res
        else:
            self.saved=x.dtype
            return res.astype(dtype)

    def backward(self,grad):
        old_dtype=self.saved
        return cp.asarray(grad,dtype=old_dtype)


### DEVICE ###

Aux=namedtuple('Aux',[
    'conv2d',
    'conv2d_bwd_x',
    'conv2d_bwd_w',
    'im2col',
    'col2im',
    'relu',
    'relu_bwd'
])
aux_cpu=Aux(
    generic_conv2d,
    generic_conv2d_bwd_x,
    generic_conv2d_bwd_w,
    _mytorch.im2col,
    _mytorch.col2im,
    generic_relu,
    generic_relu_bwd
)
aux_cuda=Aux(
    generic_conv2d,
    generic_conv2d_bwd_x,
    generic_conv2d_bwd_w,
    cuda_im2col,
    cuda_col2im,
    generic_relu,
    generic_relu_bwd
)
aux_cudnn=Aux(
    cudnn_conv2d,
    cudnn_conv2d_bwd_x,
    cudnn_conv2d_bwd_w,
    None,
    None,
    cudnn_relu,
    cudnn_relu_bwd
)

class Device:
    def __init__(self,type):
        self.type=type
        if type=='cpu':
            self.am,self.aux=np,aux_cpu
        elif type=='cuda':
            self.am=cp
            if cudnn is not None and cudnn_enabled:
                self.aux=aux_cudnn
            else:
                self.aux=aux_cuda
        else:
            raise ValueError('device type must be cpu or cuda')

    def __repr__(self): return self.type
    __str__=__repr__
    def __eq__(self,other): return self.type==other.type

def _mkdev(device=None):
    if device is None: return Device('cpu')
    elif isinstance(device,str): return Device(device)
    return device

def cuda_is_available():
    try: cp.cuda.runtime.getDeviceCount()
    except: return False
    return True


### TENSOR ###

float32=np.float32
float64=np.float64
int64=np.int64

class Tensor:
    do_grad=True
    
    def __init__(self,v,do_grad=False,dtype=None,device=None,fn=None):
        if device is None: # imply device from storage
            device=Device('cuda' if isinstance(v,cp.ndarray) else 'cpu')
        elif isinstance(device,str): device=Device(device)
        # else: device is Device object
        self.device=device
        self.am,self.aux=device.am,device.aux
        # n.b. w/ v as cupy.ndarray and device='cpu', np.asarray will
        # fail, but we'll fix it later if the need arises
        self.v=self.am.asarray(v,dtype=dtype)

        self.do_grad=do_grad
        self._grad=None
        self.fn=fn

    def __neg__(self): return NegFn()(self)
    def __mul__(self,other): return MulFn()(self,other)
    def __rmul__(self,other): return MulFn()(self,other)
    def __truediv__(self,other): return DivFn()(self,other)
    def __floordiv__(self,other): return Tensor(self.v//strip(other))
    def __rtruediv__(self,other): return RDivFn()(other,self)
    def __add__(self,other): return AddFn()(self,other)
    def __radd__(self,other): return AddFn()(self,other)
    def __sub__(self,other): return SubFn()(self,other)
    def __rsub__(self,other): return RSubFn()(other,self)

    def __isub__(self,other):
        if self.do_grad and Tensor.do_grad:
            raise TypeError('in-place operation is prohibited, since it may change the graph')
        # subtract directly, no need for SubFn here, since this op is
        # only allowed when gradient calculation is off
        self.v-=strip(other)
        return self

    def __imul__(self,other):
        if self.do_grad and Tensor.do_grad:
            raise TypeError('in-place operation is prohibited, since it may change the graph')
        # multiply directly, since gradient calculation is off
        self.v*=strip(other)
        return self

    def __abs__(self): return Tensor(abs(self.v))
    def __pow__(self,other): return PowFn()(self,other)
    def __rpow__(self,other): return RPowFn()(other,self)
    def __matmul__(self,other): return MatMulFn()(self,other)
    def __rmatmul__(self,other): return MatMulFn()(other,self)

    def __eq__(self,other): return Tensor(self.v==strip(other))
    def __ne__(self,other): return Tensor(self.v!=strip(other))
    def __le__(self,other): return Tensor(self.v<=other.v)
    def __lt__(self,other): return Tensor(self.v<strip(other))
    def __gt__(self,other): return Tensor(self.v>strip(other))
    def __bool__(self): return bool(self.v)
    
    def __repr__(self):
        r=repr(self.v).replace('array','tensor')
        if self.device.type=='cuda':
            r=r[:-1]+f", device='{self.device}')"
        if self.fn:
            r=r[:-1]+f', fn=<{self.fn.__class__.__name__}>)'
        elif self.do_grad:
            r=r[:-1]+', do_grad=True)'
        return r
    
    def __getitem__(self,key): return GetItemFn()(self,key)
    def __setitem__(self,key,val): self.v[key]=strip(val)
    def __len__(self): return len(self.v)
    def __iter__(self): return iter(self.v)

    def cos(self): return CosFn()(self)
    def sin(self): return SinFn()(self)
    def sqrt(self): return PowFn()(self,.5)
    def sum(self,**kws): return SumFn()(self,**kws)
    def abs(self): return abs(self)

    def mean(self,axis=None,**kws):
        v=self.v
        if axis is None:
            n=v.size
        elif isinstance(axis,tuple):
            n=1
            for a in axis:
                n*=v.shape[a]
        else:
            n=v.shape[axis]
        # It seems numpy converts n to float64 when dividing by it. In
        # cases where sum is reduced to a scalar this may promote the
        # (e.g. float32) result itself to float64. To keep original
        # dtype convert n to it explicitly. Done only for float
        # types. This is the same behavior as np.mean.
        if np.issubdtype(v.dtype,np.floating): n=v.dtype.type(n)
        return self.sum(axis=axis,**kws)/n

    def var(self): return Tensor(self.v.var())
    def std(self): return Tensor(self.v.std())

    def exp(self): return ExpFn()(self)
    def log(self): return LogFn()(self)
    def sigmoid(self): return SigmoidFn()(self)
    def reshape(self,shape): return ReshapeFn()(self,shape)
    def argmax(self,**kws): return Tensor(self.v.argmax(**kws))

    def argsort(self,*args,**kws):
        return Tensor(self.v.argsort(*args,**kws))

    def max(self): return Tensor(self.v.max())
    def min(self): return Tensor(self.v.min())
    def all(self): return Tensor(self.v.all())

    def histc(self,bins=10,min=0,max=0):
        bounds=None
        if not min==max==0: bounds=(min,max)
        return Tensor(self.am.histogram(self.v,bins=bins,range=bounds)[0])

    def histogram(self,*args,**kws):
        hist,edges=self.am.histogram(self.v,*args,**kws)
        return Tensor(hist),Tensor(edges)
    
    def zero_(self):
        if self.do_grad and Tensor.do_grad:
            raise TypeError('in-place operation is prohibited, since it may change the graph')
        # zero all elements, this works faster than creating new array
        # w/ zeros_like()
        self.v[...]=0
        return self

    def to(self,dtype=None,device=None):
        old_dev,old_dt=self.device,self.v.dtype
        new_dev=old_dev if device is None else Device(device)
        new_dt=old_dt if dtype is None else dtype
        if new_dev==old_dev:
            if new_dt==old_dt: return self
            fn=DTypeFn
        else:
            fn=CPUFn if new_dev.type=='cpu' else CUDAFn
        return fn()(self,dtype)

    @no_grad()
    def to_(self,dtype=None,device=None):
        t=self.to(dtype=dtype,device=device)
        if t is self: return self
        self.v=t.v
        self.device=t.device
        self.am=t.am
        self.aux=t.aux
        if self._grad is not None:
            self._grad.to_(dtype=dtype,device=device)
        return self

    def float(self): return self.to(dtype=float32)
    def cuda(self): return self.to(device='cuda')
    def cpu(self): return self.to(device='cpu')

    def new_tensor(self,v,do_grad=False):
        return Tensor(v,dtype=self.v.dtype,device=self.device,
                      do_grad=do_grad)

    def backward(self,*args,**kws):
        if not self.do_grad: raise TypeError("this tensor doesn't require gradients")
        backward(self,*args,**kws)

    def do_grad_(self,do_grad=True):
        self.do_grad=do_grad
        return self

    # n.b. in torch this op is recorded in the graph, so grads coming
    # to cloned tensor, also come to original one
    def clone(self):
        t=Tensor(self.v.copy(),do_grad=self.do_grad,fn=self.fn)
        if self._grad is not None: t._grad=Tensor(self._grad.v.copy())
        return t

    def detach_(self):
        self.do_grad=False
        self._grad=None
        self.fn=None
        return self

    def detach(self): return Tensor(self.v)
    def item(self): return self.v.item()

    @property
    def grad(self): return self._grad
    @grad.setter
    def grad(self,other): self._grad=other
    
    @property
    def shape(self): return self.v.shape
    @property
    def dtype(self): return self.v.dtype
    @property
    def T(self): return Tensor(self.v.T)
    @property
    def ndim(self): return self.v.ndim


### CREATION ###

def strip(t): return t.v if isinstance(t,Tensor) else t

def iterstrip(t):
    try: iter(t)
    except TypeError: return strip(t)
    return [strip(el) for el in t]

def tensor(v,**kws): return Tensor(iterstrip(v),**kws)

def empty(shape,dtype=None,do_grad=False,device=None):
    dev=_mkdev(device)
    return Tensor(dev.am.empty(shape,dtype=dtype),
                  do_grad=do_grad,device=dev)

def full(shape,fill_value,dtype=None,do_grad=False,device=None):
    dev=_mkdev(device)
    return Tensor(dev.am.full(shape,fill_value,dtype=dtype),
                  do_grad=do_grad,device=dev)

def zeros(shape,dtype=None,do_grad=False,device=None):
    dev=_mkdev(device)
    return Tensor(dev.am.zeros(shape,dtype=dtype),
                  do_grad=do_grad,device=dev)

def zeros_like(t,dtype=None,do_grad=False,device=None):
    if dtype is None: dtype=t.v.dtype
    if device is None: device=t.device
    return zeros(t.v.shape,dtype=dtype,do_grad=do_grad,device=device)

def ones(shape,dtype=None,do_grad=False,device=None):
    dev=_mkdev(device)
    return Tensor(dev.am.ones(shape,dtype=dtype),
                  do_grad=do_grad,device=dev)

def ones_like(t,dtype=None,do_grad=False,device=None):
    if dtype is None: dtype=t.v.dtype
    if device is None: device=t.device
    return ones(t.v.shape,dtype=dtype,do_grad=do_grad,device=device)

def arange(*args,dtype=None,do_grad=False,device=None):
    dev=_mkdev(device)
    return Tensor(dev.am.arange(*args,dtype=dtype),
                  do_grad=do_grad,device=dev)

def linspace(*args,dtype=None,do_grad=False,device=None,**kws):
    dev=_mkdev(device)
    return Tensor(dev.am.linspace(*args,**kws,dtype=dtype),
                  do_grad=do_grad,device=dev)


### RNG ###

# We don't use global rng funcs from cupy, since it doesn't provide
# random.RandomState and we don't want to mess its global state from
# inside this library. So to get a random array on gpu it's first
# generated w/ our private numpy's RandomState and then moved to gpu
# inside Tensor constructor by means of the device arg. It must be
# slower than direct generation on gpu, but the upside is that we have
# same number sequences on cpu and gpu when seeded the same.

rs=np.random.RandomState()

def manual_seed(seed): rs.seed(seed)

def randn(*args,dtype=None,do_grad=False,device=None):
    return Tensor(rs.randn(*args),dtype=dtype,
                  do_grad=do_grad,device=device)

def randn_like(t,dtype=None,do_grad=False,device=None):
    if dtype is None: dtype=t.v.dtype
    if device is None: device=t.device
    return randn(*t.v.shape,dtype=dtype,do_grad=do_grad,device=device)

def rand(*args,dtype=None,do_grad=False,device=None):
    return Tensor(rs.rand(*args),dtype=dtype,
                  do_grad=do_grad,device=device)

def normal(mean,std,size,dtype=None,do_grad=False,device=None):
    return Tensor(rs.normal(mean,std,size),dtype=dtype,
                  do_grad=do_grad,device=device)

def randperm(n,dtype=None,do_grad=False,device=None):
    return Tensor(rs.permutation(n),dtype=dtype,
                  do_grad=do_grad,device=device)


### FUNCTIONAL ###

def log_softmax(x): return LogSoftmaxFn()(x)
def linear(x,w,b=None): return LinearFn()(x,w,b)
def relu(x): return ReLUFn()(x)

def conv2d(x,w,b=None,stride=1,padding=0):
    return Conv2dFn()(x,w,b,stride=stride,padding=padding)

def maxpool2d(x,ksize,stride=1,padding=0):
    return MaxPool2dFn()(x,ksize,stride=stride,padding=padding)

def nll_loss(x,targ):
    n=len(x)
    return -x[x.am.arange(n),targ.v].sum()/x.v.dtype.type(n)

def cross_entropy(x,targ): return nll_loss(log_softmax(x),targ)


### NN.INIT ###

def kaiming_normal_(t):
    assert t.ndim>=2
    if t.ndim==2:
        fan_in=t.shape[0]
    else:
        fan_in=1
        for n in t.shape[1:]:
            fan_in*=n

    std=(2./fan_in)**.5
    t.v=normal(0.,std,t.shape,dtype=t.dtype,device=t.device).v
    return t


### NN ###

class Module:
    def __init__(self):
        self._modules=[]
        self.hooks=[]
        self.extra_repr=''

    def params(self):
        p=[]
        for m in [self]+self._modules:
            if hasattr(m,'w'):
                p.append(m.w)
                if m.b is not None: p.append(m.b)
        return p

    def do_grad_(self,do_grad=True):
        for p in self.params(): p.do_grad_(do_grad)
        return self

    def to(self,dtype=None,device=None):
        for p in self.params():
            p.to_(dtype=dtype,device=device)
        return self

    def cuda(self,device=None):
        if device is None: device='cuda'
        return self.to(device=device)

    def cpu(self): return self.to(device='cpu')

    def __call__(self,x):
        out=self.forward(x)
        if self.hooks: [h(self,x,out) for h in self.hooks]
        return out

    def add_forward_hook(self,hook):
        self.hooks.append(hook)

    def __getitem__(self,key): return self._modules[key]
    def __len__(self): return len(self._modules)

    def __repr__(self):
        lines=[]
        for i,m in enumerate(self._modules):
            mlines=repr(m).split('\n')
            mlines[0]='%s: %s' % (i,mlines[0])
            lines.extend(mlines)

        extra=self.extra_repr
        # currently parens are for children OR for extra
        assert not (lines and extra)
        s=self.__class__.__name__
        if extra: s+='('+extra+')'
        if lines: s+='(\n'+'\n'.join(['  '+l for l in lines])+'\n)'
        return s

    __str__=__repr__

class Linear(Module):
    def __init__(self,n_in,n_out,bias=True):
        super().__init__()
        self.w=kaiming_normal_(empty((n_in,n_out),do_grad=True))
        self.b=None
        if bias: self.b=zeros((1,n_out),do_grad=True)
        self.extra_repr='n_in=%d, n_out=%d, bias=%s' % (n_in,n_out,bias)

    def forward(self,x): return linear(x,self.w,self.b)

class Sigmoid(Module):
    def forward(self,x): return x.sigmoid()

class ReLU(Module):
    def forward(self,x): return relu(x)

class Softmax(Module):
    def forward(self,x):
        e=x.exp()
        return e/e.sum(axis=1,keepdims=True)

class LogSoftmax(Module):
    def forward(self,x): return log_softmax(x)

class Seq(Module):
    def __init__(self,*modules):
        super().__init__()
        self._modules=list(modules)

    def forward(self,x):
        for m in self._modules:
            x=m(x)
        return x

class Conv2d(Module):
    def __init__(self,ch_in,ch_out,ksize,stride=1,padding=0):
        super().__init__()
        self.stride,self.padding=stride,padding
        self.w=kaiming_normal_(empty((ch_out,ch_in,ksize,ksize),do_grad=True))
        self.b=zeros(ch_out,do_grad=True)
        self.extra_repr='ch_in=%d, ch_out=%d, ksize=%d, stride=%d, padding=%d' % (
            ch_in,ch_out,ksize,stride,padding)

    def forward(self,x):
        return conv2d(x,self.w,self.b,stride=self.stride,
                      padding=self.padding)

class MaxPool2d(Module):
    def __init__(self,ksize,stride=1,padding=0):
        super().__init__()
        self.ksize,self.stride,self.padding=ksize,stride,padding

    def forward(self,x):
        return maxpool2d(x,self.ksize,self.stride,self.padding)


### DATA ###

class TensorDataset:
    def __init__(self,*ts):
        for t in ts:
            assert t.shape[0]==ts[0].shape[0],'tensors must be of the same shape'
        self.ts=ts

    def __getitem__(self,key): return tuple(t[key] for t in self.ts)
    def __len__(self): return len(self.ts[0])

class DataLoader:
    def __init__(self,ds,bs=1,shuffle=False):
        self.ds,self.bs,self.shuffle=ds,bs,shuffle

    def __iter__(self):
        n=len(self.ds)
        if self.shuffle: idxs=randperm(n)
        else: idxs=arange(n)
        for i in range(0,n,self.bs):
            yield self.ds[idxs[i:i+self.bs].v]

    def __len__(self): return len(self.ds)


### OPTIM ###

class SGD:
    def __init__(self,params,lr,l2_decay=0.,zero_grad=True):
        self.params,self.lr,self.l2_decay=params,lr,l2_decay
        self._zero_grad=zero_grad

    @no_grad()
    def step(self):
        for p in self.params:
            if self.l2_decay>0.: p*=1.-self.l2_decay
            p-=self.lr*p.grad
        if self._zero_grad: self.zero_grad()

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()
