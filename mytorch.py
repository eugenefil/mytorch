import numpy as np

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
def backward(t,create_graph=False):
    # since grad calculation is internal stuff, during graph
    # traversal running gradient is stored as numpy array instead
    # of Tensor object to avoid extra funcalls, but at leaves
    # numpy is converted to Tensor
    lst=[(t,t.v.dtype.type(1))] # (tensor,running gradient)
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

    def __pow__(self,other): return PowFn()(self,other)
    def __rpow__(self,other): return RPowFn()(other,self)
    def __matmul__(self,other): return MatMulFn()(self,other)
    def __rmatmul__(self,other): return MatMulFn()(other,self)

    def __eq__(self,other): return Tensor(self.v==other.v)
    def __le__(self,other): return Tensor(self.v<=other.v)
    def __lt__(self,other): return Tensor(self.v<other.v)
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

    def exp(self): return ExpFn()(self)
    def log(self): return LogFn()(self)
    def sigmoid(self): return SigmoidFn()(self)
    def log_softmax(self): return LogSoftmaxFn()(self)
    def reshape(self,shape): return Tensor(self.v.reshape(shape))
    def argmax(self,**kws): return Tensor(self.v.argmax(**kws))
    
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

    def backward(self,**kws):
        if not self.do_grad: raise TypeError("this tensor doesn't require gradients")
        backward(self,**kws)

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


def strip(t): return t.v if isinstance(t,Tensor) else t
def iterstrip(t):
    try: iter(t)
    except TypeError: return strip(t)
    return [strip(el) for el in t]

def zeros(shape,do_grad=False):
    return Tensor(np.zeros(shape),do_grad=do_grad)
def zeros_like(*args,**kws): return Tensor(np.zeros_like(*args,**kws))
def ones(shape): return Tensor(np.ones(shape))
def arange(*args,**kws): return Tensor(np.arange(*args,**kws))
def randn(*args,do_grad=False):
    return Tensor(rs.randn(*args),do_grad=do_grad)
def randperm(n): return Tensor(rs.permutation(n))
def linspace(*args): return Tensor(np.linspace(*args))
def linear(x,w,b=None): return LinearFn()(x,w,b)
def tensor(v,**kws): return Tensor(iterstrip(v),**kws)

def manual_seed(seed): rs.seed(seed)


class Module:
    def params(self):
        p=[]
        for m in self.modules:
            if hasattr(m,'w'):
                p.append(m.w)
                if m.b is not None: p.append(m.b)
        return p

    def to(self,dtype):
        for p in self.params():
            p.to_(dtype)
        return self

class Linear(Module):
    def __init__(self,n_in,n_out,bias=True):
        self.modules=[self]
        self.w=randn(n_in,n_out,do_grad=True)
        self.b=None
        if bias: self.b=zeros((1,n_out),do_grad=True)

    def __call__(self,x): return linear(x,self.w,self.b)

class Sigmoid:
    def __call__(self,x): return x.sigmoid()

class Softmax:
    def __call__(self,x):
        e=x.exp()
        return e/e.sum(axis=1,keepdims=True)

class LogSoftmax:
    def __call__(self,x): return x.log_softmax()

class Seq(Module):
    def __init__(self,*modules):
        self.modules=modules

    def __call__(self,x):
        for m in self.modules:
            x=m(x)
        return x
