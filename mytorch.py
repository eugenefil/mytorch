import numpy as np

rs=np.random.RandomState()

class Fn:
    def __call__(self,*args,**kws):
        res=self.forward(*args,**kws)
        if Tensor.do_grad:
            if len(args)==1:
                res.do_grad=args[0].do_grad
            else:
                self.needs_grad=[isinstance(a,Tensor) and a.do_grad
                                 for a in args]
                res.do_grad=any(self.needs_grad)
            if res.do_grad:
                self.args=args
                res.fn=self
        return res
    
    def _backward(self,grad):
        return self.backward(grad,*self.args)

class NegFn(Fn):
    def forward(self,x): return Tensor(-x.v)
    def backward(self,grad,x): return -grad
    
class AddFn(Fn):
    def forward(self,x,y): return Tensor(x.v+strip(y))
    def backward(self,grad,x,y): return [grad,grad]
    
class SubFn(Fn):
    def forward(self,x,y): return Tensor(x.v-strip(y))
    def backward(self,grad,x,y): return [
            grad,
            -grad if self.needs_grad[1] else None
    ]
    
class RSubFn(SubFn):
    def forward(self,x,y): return Tensor(strip(x)-y.v)

class MulFn(Fn):
    def forward(self,x,y): return Tensor(x.v*strip(y))
    def backward(self,grad,x,y): return [
            grad*y if self.needs_grad[0] else None,
            grad*x if self.needs_grad[1] else None
    ]
    
class DivFn(Fn):
    def forward(self,x,y): return Tensor(x.v/strip(y))
    def backward(self,grad,x,y): return [
            grad/y if self.needs_grad[0] else None,
            -grad*x/y**2. if self.needs_grad[1] else None
    ]
    
class RDivFn(DivFn):
    def forward(self,x,y): return Tensor(strip(x)/y.v)

class PowFn(Fn):
    def forward(self,x,y): return Tensor(x.v**strip(y))
    # don't show warning for neg/zero log argument, just return nan
    @np.errstate(invalid='ignore',divide='ignore')
    def backward(self,grad,x,y): return [
            grad*y*x**(y-1.) if self.needs_grad[0] else None,
            grad*x**y*log(x) if self.needs_grad[1] else None
    ]

class ExpFn(Fn):
    def forward(self,x):
        self.res=Tensor(np.exp(x.v))
        return self.res
    def backward(self,grad,x): return grad*self.res

class LogFn(Fn):
    def forward(self,x): return Tensor(np.log(x.v))
    def backward(self,grad,x): return grad/x
    
class MatMulFn(Fn):
    def forward(self,x,y): return Tensor(x.v@y.v)
    def backward(self,grad,x,y): return [
            grad@y.T if self.needs_grad[0] else None,
            x.T@grad if self.needs_grad[1] else None
    ]

class GetItemFn(Fn):
    def forward(self,x,key): return Tensor(x.v[key])
    def backward(self,grad,x,key):
        out=zeros(x.shape)
        out[key]=grad
        return [out,None]
    
class SumFn(Fn):
    def forward(self,x,axis=None,keepdims=False):
        self.axis=None
        # if axes are reduced, store info to correctly broadcast grads
        # to original shape later
        if not keepdims:
            self.axis=tuple(range(x.ndim)) if axis is None else axis
        return Tensor(np.sum(x.v,axis=axis,keepdims=keepdims))

    def backward(self,grad,x):
        grad=grad.v
        # if axes were reduced, restore them to broadcast grad correctly
        if self.axis is not None:
            grad=np.expand_dims(grad,self.axis)
        return Tensor(np.broadcast_to(grad,x.shape))

class CosFn(Fn):
    def forward(self,x): return Tensor(np.cos(x.v))
    def backward(self,grad,x): return -grad*x.sin()
    
class SinFn(Fn):
    def forward(self,x): return Tensor(np.sin(x.v))
    def backward(self,grad,x): return grad*x.cos()


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


class Tensor:
    do_grad=True
    
    def __init__(self,v,do_grad=False,fn=None):
        self.v=np.asarray(v)
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
        self.v=SubFn()(self,other).v
        return self

    def __pow__(self,other): return PowFn()(self,other)
    def __rpow__(self,other): return PowFn()(other,self)
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
        if ax is None: ax=tuple(range(self.ndim))
        if not isinstance(ax,tuple): ax=(ax,)
        n=1
        for a in ax:
            n*=self.shape[a]
        return self.sum(axis=axis,**kws)/n

    def exp(self): return ExpFn()(self)
    def log(self): return LogFn()(self)
    def sigmoid(self): return 1./(1.+(-self).exp())
    def reshape(self,shape): return Tensor(self.v.reshape(shape))
    def argmax(self,**kws): return Tensor(self.v.argmax(**kws))
    
    def zero_(self):
        if self.do_grad and Tensor.do_grad:
            raise TypeError('in-place operation is prohibited, since it may change the graph')
        self.v=zeros(self.shape).v
        return self

    @no_grad()
    def backward(self,create_graph=False):
        if not self.do_grad: raise TypeError("this tensor doesn't require gradients")
        lst=[(self,Tensor(1.))]
        while lst:
            t,tgrad=lst.pop()
            # if tensor was broadcasted, so grad has different shape,
            # sum-reduce grad to original tensor shape
            if tgrad.shape!=t.shape:
                ddim=tgrad.ndim-t.ndim
                assert ddim>=0, "broadcasting can't decrease num of dims"
                bcast=(1,)*ddim+t.shape if ddim>0 else t.shape
                axes=tuple([
                    i for i,(ng,nt) in enumerate(zip(tgrad.shape,bcast))
                    if ng>nt
                ])
                # sum-reduce axes that were broadcasted
                if axes: tgrad=tgrad.sum(axis=axes,keepdims=True)
                # if broadcasting added axes, reshape to original
                if ddim>0: tgrad=tgrad.reshape(t.shape)

            if not t.fn or create_graph: # if leaf or saving grad to every node
                if t.grad is None:
                    t.grad=tgrad
                else:
                    t.grad+=tgrad

            if t.fn:
                if len(t.fn.args)==1:
                    lst.append((t.fn.args[0],t.fn._backward(tgrad)))
                else:
                    lst.extend([
                        (arg,grad)
                        for arg,needs_grad,grad in
                        zip(t.fn.args,t.fn.needs_grad,t.fn._backward(tgrad))
                        if needs_grad
                    ])

    @property
    def grad(self): return self._grad
    @grad.setter
    def grad(self,other): self._grad=other
    
    @property
    def shape(self): return self.v.shape
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
def ones(shape): return Tensor(np.ones(shape))
def arange(*args,**kws): return Tensor(np.arange(*args,**kws))
def randn(*args,do_grad=False):
    return Tensor(rs.randn(*args),do_grad=do_grad)
def randperm(n): return Tensor(rs.permutation(n))
def linspace(*args): return Tensor(np.linspace(*args))
def log(t): return Tensor(np.log(t.v))
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

class Linear(Module):
    def __init__(self,n_in,n_out,bias=True):
        self.modules=[self]
        self.w=randn(n_in,n_out,do_grad=True)
        self.b=None
        if bias: self.b=zeros((1,n_out),do_grad=True)

    def __call__(self,x):
        z=x@self.w
        if self.b is not None: z+=self.b
        return z

class Sigmoid:
    def __call__(self,x): return x.sigmoid()

class Softmax:
    def __call__(self,x):
        e=x.exp()
        return e/e.sum(axis=1,keepdims=True)

class LogSoftmax:
    def __call__(self,x):
        return x-x.exp().sum(axis=1,keepdims=True).log()

class Seq(Module):
    def __init__(self,*modules):
        self.modules=modules

    def __call__(self,x):
        for m in self.modules:
            x=m(x)
        return x
