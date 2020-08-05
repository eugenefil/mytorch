import numpy as np

import _mytorch

rs=np.random.RandomState()

float32=np.float32
float64=np.float64
int64=np.int64


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
    # since grad calculation is internal stuff, during graph
    # traversal running gradient is stored as numpy array instead
    # of Tensor object to avoid extra funcalls, but at leaves
    # numpy is converted to Tensor
    if grad is None: grad=np.ones_like(t.v)
    else: grad=np.asarray(strip(grad))
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
        x,y=x.v,strip(y)
        self.saved=(x,y)
        return x**y

    def backward(self,grad):
        x,y=self.saved
        return (
            grad*y*x**(y-1.) if self.needs_grad[0] else None,
            grad*x**y*np.log(x) if self.needs_grad[1] else None
        )

class RPowFn(Fn):
    def forward(self,x,y):
        self.args=(y,) # only y may be a tensor
        res=x**y.v
        self.saved=(x,res)
        return res

    def backward(self,grad):
        x,x_pow_y=self.saved
        return grad*x_pow_y*np.log(x)

class ExpFn(Fn):
    def forward(self,x):
        self.saved=np.exp(x.v)
        return self.saved
    def backward(self,grad): return grad*self.saved

class LogFn(Fn):
    def forward(self,x):
        self.saved=x.v
        return np.log(self.saved)
    def backward(self,grad): return grad/self.saved

class SigmoidFn(Fn):
    def forward(self,x):
        x=x.v
        # cast 1. to x.dtype to keep original type, otherwise numpy
        # will promote 1. (and the end result) to float64 when x is a
        # scalar
        one=x.dtype.type(1)
        res=one/(one+np.exp(-x))
        self.saved=(one,res)
        return res

    def backward(self,grad):
        one,res=self.saved
        return grad*res*(one-res)

class ReLUFn(Fn):
    def forward(self,x):
        self.saved=np.maximum(x.v,0.)
        return self.saved

    def backward(self,grad):
        grad=grad.copy()
        grad[self.saved==0.]=0.
        return grad

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
        x=x.v
        z=x-x.max(axis=1,keepdims=True)
        ez=np.exp(z)
        ezsum=ez.sum(axis=1,keepdims=True)
        self.saved=(ez,ezsum)
        return z-np.log(ezsum)

    def backward(self,grad):
        ez,ezsum=self.saved
        return grad-ez/ezsum*grad.sum(axis=1,keepdims=True)

class MatMulFn(Fn):
    def forward(self,x,y):
        x,y=x.v,y.v
        self.saved=(x,y)
        return np.matmul(x,y)

    def backward(self,grad):
        x,y=self.saved
        return (
            np.matmul(grad,y.T) if self.needs_grad[0] else None,
            np.matmul(x.T,grad) if self.needs_grad[1] else None
        )

class GetItemFn(Fn):
    # W/ advanced indexing the key could be a tuple of tensors, in
    # which case we'd like to make a stripped tuple of numpy arrays
    # from it for passing to numpy. But we don't do this. To get
    # indices from each tensor numpy creates an iter, which per our
    # implementation (see Tensor.__iter__) is a native numpy iter. So
    # we're fine w/out making a stripped tuple.
    def forward(self,x,key):
        self.args=(x,)
        x=x.v
        self.saved=(x,key)
        return x[key]

    def backward(self,grad):
        x,key=self.saved
        out=np.zeros_like(x)
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
        x=x.v
        self.saved=(x,axis,keepdims)
        # note: x.sum() is faster than np.sum(x)
        return x.sum(axis=axis,keepdims=keepdims)

    def backward(self,grad):
        x,axis,keepdims=self.saved
        # if axes were reduced, restore to broadcast grad correctly
        if not keepdims:
            axis=tuple(range(x.ndim)) if axis is None else axis
            grad=np.expand_dims(grad,axis)
        return np.broadcast_to(grad,x.shape)

class CosFn(Fn):
    def forward(self,x):
        self.saved=x.v
        return np.cos(self.saved)
    def backward(self,grad): return -grad*np.sin(self.saved)

class SinFn(Fn):
    def forward(self,x):
        self.saved=x.v
        return np.sin(self.saved)
    def backward(self,grad): return grad*np.cos(self.saved)

# Here we bundle x@w+b into a single op. This saves a graph node and
# some calculations. Backward formulas are taken from MatMulFn and
# AddFn. Also knowing that bias would be broadcasted in a certain way
# avoids reducing that would otherwise be done in a general way in
# Tensor.backward(). Using this custom op instead of general code gave
# 5% reduction in time.
class LinearFn(Fn):
    def forward(self,x,w,b):
        x,w=x.v,w.v
        z=np.matmul(x,w)
        if b is not None: z+=b.v
        self.saved=(x,w)
        return z

    def backward(self,grad):
        x,w=self.saved
        return (
            np.matmul(grad,w.T) if self.needs_grad[0] else None,
            np.matmul(x.T,grad) if self.needs_grad[1] else None,
            grad.sum(axis=0,keepdims=True) if self.needs_grad[2] else None
        )

class Conv2dFn(Fn):
    def forward(self,x,w,b,stride=1,padding=0):
        x,w=x.v,w.v
        ch_out,ch_in,ksize_h,ksize_w=w.shape
        c=ch_in*ksize_h*ksize_w
        wc=w.reshape((ch_out,c))

        n,ch_in1,h_in,w_in=x.shape
        assert ch_in1==ch_in # match channels between input and weights

        h_out=(h_in+2*padding-ksize_h)//stride+1
        w_out=(w_in+2*padding-ksize_w)//stride+1
        # (n,ch_in,h_in,w_in) -> (n,c,h_out*w_out)
        xk=_mytorch.extract_kernels(x,ksize_h,ksize_w,stride,padding)
        self.saved=(xk,wc,w.shape,x.shape,stride,padding)
        # bcast wc (ch_out,c) -> (n,ch_out,c),
        # (n,ch_out,c) @ (n,c,h_out*w_out) = (n,ch_out,h_out*w_out)
        res=np.matmul(wc,xk)

        if b is not None:
            b=b.v
            assert b.shape==(ch_out,)
            b=b.reshape((ch_out,1))
            # bcast b (ch_out,1) -> (n,ch_out,h_out*w_out)
            res+=b

        return res.reshape(n,ch_out,h_out,w_out)

    def backward(self,grad):
        xk,wc,w_shape,x_shape,stride,padding=self.saved
        # flatten channels (n,ch_out,h_out,w_out) -> (n,ch_out,h_out*w_out)
        grad=grad.reshape(grad.shape[:2]+(-1,))
        if self.needs_grad[0]:
            # bcast wc.T (c,ch_out) -> (n,c,ch_out),
            # (n,c,ch_out) @ (n,ch_out,h_out*w_out) = (n,c,h_out*w_out)
            xk_grad=np.matmul(wc.T,grad)
            ch_in,h_in,w_in=x_shape[1:]
            ksize_h,ksize_w=w_shape[2:]
            # (n,c,h_out*w_out) -> (n,ch_in,h_in,w_in)
            x_grad=_mytorch.extract_kernels_backward(xk_grad,
                                                     ch_in,h_in,w_in,
                                                     ksize_h,ksize_w,
                                                     stride,padding)

        if self.needs_grad[1]:
            # transpose xk (n,c,h_out*w_out) -> (n,h_out*w_out,c),
            # (n,ch_out,h_out*w_out) @ (n,h_out*w_out,c) = (n,ch_out,c),
            # sum (n,ch_out,c) -> (ch_out,c)
            wc_grad=np.matmul(grad,xk.transpose(0,2,1)).sum(axis=0)
            # (ch_out,c) -> (ch_out,ch_in,ksize_h,ksize_w)
            w_grad=wc_grad.reshape(w_shape)

        return (
            x_grad if self.needs_grad[0] else None,
            w_grad if self.needs_grad[1] else None,
            # (n,ch_out,h_out*w_out) -> (ch_out,)
            grad.sum(axis=(0,2)) if self.needs_grad[2] else None
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


class Tensor:
    do_grad=True
    
    def __init__(self,v,do_grad=False,dtype=None,fn=None):
        self.v=np.asarray(v,dtype=dtype)
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
        if self.fn:
            r=r[:-1]+f', fn=<{self.fn.__class__.__name__}>)'
        elif self.do_grad:
            r=r[:-1]+f', do_grad=True)'
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
        ax=axis
        if ax is None: ax=tuple(range(self.v.ndim))
        if not isinstance(ax,tuple): ax=(ax,)
        n=1
        for a in ax:
            n*=self.v.shape[a]
        # It seems numpy converts n to float64 when dividing by it. In
        # cases where sum is reduced to a scalar this may promote the
        # (e.g. float32) result itself to float64. To keep original
        # dtype convert n to it explicitly. Done only for float
        # types. This is the same behavior as np.mean.
        if np.issubdtype(self.v.dtype,np.floating):
            n=self.v.dtype.type(n)
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
        return Tensor(np.histogram(self.v,bins=bins,range=bounds)[0])

    def histogram(self,*args,**kws):
        return (*map(tensor,np.histogram(self.v,*args,**kws)),)
    
    def zero_(self):
        if self.do_grad and Tensor.do_grad:
            raise TypeError('in-place operation is prohibited, since it may change the graph')
        # zero all elements, this works faster than creating new array
        # w/ zeros_like()
        self.v[...]=0
        return self

    def to_(self,dtype):
        self.v=np.asarray(self.v,dtype=dtype)
        if self._grad is not None: self._grad.to_(dtype)
        return self

    def to(self,dtype):
        v=np.asarray(self.v,dtype=dtype)
        if v is self.v: return self

        t=Tensor(v,do_grad=self.do_grad)
        if self._grad is not None: t._grad=self._grad.to(dtype)
        return t

    def float(self): return self.to(float32)

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


def tensor(v,**kws): return Tensor(iterstrip(v),**kws)
def strip(t): return t.v if isinstance(t,Tensor) else t
def iterstrip(t):
    try: iter(t)
    except TypeError: return strip(t)
    return [strip(el) for el in t]

def empty(shape,do_grad=False):
    return Tensor(np.empty(shape),do_grad=do_grad)

def zeros(shape,do_grad=False):
    return Tensor(np.zeros(shape),do_grad=do_grad)

def zeros_like(*args,**kws): return Tensor(np.zeros_like(*args,**kws))
def ones(shape,do_grad=False):
    return Tensor(np.ones(shape),do_grad=do_grad)
def arange(*args,dtype=None,do_grad=False):
    return Tensor(np.arange(*args,dtype=dtype),do_grad=do_grad)

def randn(*args,do_grad=False):
    return Tensor(rs.randn(*args),do_grad=do_grad)
def randn_like(t): return randn(*t.v.shape)

def normal(mean,std,size,do_grad=False):
    return Tensor(rs.normal(mean,std,size),do_grad=do_grad)

def randperm(n): return Tensor(rs.permutation(n))
def linspace(*args): return Tensor(np.linspace(*args))

def log_softmax(x): return LogSoftmaxFn()(x)
def linear(x,w,b=None): return LinearFn()(x,w,b)
def relu(x): return ReLUFn()(x)

def conv2d(x,w,b=None,stride=1,padding=0):
    return Conv2dFn()(x,w,b,stride=stride,padding=padding)

def maxpool2d(x,ksize,stride=1,padding=0):
    return MaxPool2dFn()(x,ksize,stride=stride,padding=padding)

def nll_loss(x,targ):
    n=len(x)
    return -x[arange(n),targ].sum()/x.dtype.type(n)

def cross_entropy(x,targ): return nll_loss(log_softmax(x),targ)

def manual_seed(seed): rs.seed(seed)


def kaiming_normal_(t):
    assert t.ndim>=2
    if t.ndim==2:
        fan_in=t.shape[0]
    else:
        fan_in=1
        for n in t.shape[1:]:
            fan_in*=n

    std=np.sqrt(2./fan_in)
    t.v=rs.normal(0.,std,t.shape)
    return t


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

    def to(self,dtype):
        for p in self.params():
            p.to_(dtype)
        return self

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
