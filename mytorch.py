import numpy as np

rs=np.random.RandomState()

class Fn:
    def __call__(self,*args,**kws):
        res=self.forward(*args,**kws)
        if Tensor.do_grad:
            self.args=args
            res.do_grad=sum([isinstance(a,Tensor) and a.do_grad
                             for a in args])>0
            res.fn=self if res.do_grad else None
        return res
    
    def _backward(self,grad):
        return self.backward(grad,*self.args)

class NegFn(Fn):
    def forward(self,x): return Tensor(-Tensor.strip(x))
    def backward(self,grad,x): return [-grad]
    
class AddFn(Fn):
    def forward(self,x,y): return Tensor(Tensor.strip(x)+Tensor.strip(y))
    def backward(self,grad,x,y): return [grad,grad]
    
class SubFn(Fn):
    def forward(self,x,y): return Tensor(Tensor.strip(x)-Tensor.strip(y))
    def backward(self,grad,x,y): return [grad,-grad]
    
class MulFn(Fn):
    def forward(self,x,y): return Tensor(Tensor.strip(x)*Tensor.strip(y))
    def backward(self,grad,x,y): return [grad*y,grad*x]
    
class DivFn(Fn):
    def forward(self,x,y): return Tensor(Tensor.strip(x)/Tensor.strip(y))
    def backward(self,grad,x,y): return [grad/y,-grad*x/y**2.]
    
class PowFn(Fn):
    def forward(self,x,y): return Tensor(Tensor.strip(x)**Tensor.strip(y))
    # don't show warning for neg/zero log argument, just return nan
    @np.errstate(invalid='ignore',divide='ignore')
    def backward(self,grad,x,y):
        return [grad*y*x**(y-1.),grad*x**y*log(x)]

class ExpFn(Fn):
    def forward(self,x):
        self.res=Tensor(np.exp(Tensor.strip(x)))
        return self.res
    def backward(self,grad,x): return [grad*self.res]
    
class MatMulFn(Fn):
    def forward(self,x,y):
        return Tensor(np.matmul(Tensor.strip(x),Tensor.strip(y)))
    def backward(self,grad,x,y): return [grad@y.T,x.T@grad]

class GetItemFn(Fn):
    def forward(self,x,key): return Tensor(Tensor.strip(x)[key])
    def backward(self,grad,x,key):
        out=zeros(x.shape)
        out[key]=grad
        return [out,0.] # grad for `key' argument is 0
    
class SumFn(Fn):
    def forward(self,x,**kws):
        return Tensor(np.sum(Tensor.strip(x),**kws))
    def backward(self,grad,x): return [grad*ones(x.shape)]

class CosFn(Fn):
    def forward(self,x): return Tensor(np.cos(Tensor.strip(x)))
    def backward(self,grad,x): return [-grad*x.sin()]
    
class SinFn(Fn):
    def forward(self,x): return Tensor(np.sin(Tensor.strip(x)))
    def backward(self,grad,x): return [grad*x.cos()]


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
        self.v=np.array(Tensor.strip(v))
        self.do_grad=do_grad
        self._grad=None
        self.fn=fn

    @staticmethod
    def strip(t):
        try:
            iter(t)
        except TypeError:
            return t.v if isinstance(t,Tensor) else t
        return np.array([el.v if isinstance(el,Tensor) else el for el in t])

    def __neg__(self): return NegFn()(self)
    def __mul__(self,other): return MulFn()(self,other)
    def __rmul__(self,other): return MulFn()(self,other)
    def __truediv__(self,other): return DivFn()(self,other)
    def __rtruediv__(self,other): return DivFn()(other,self)
    def __add__(self,other): return AddFn()(self,other)
    def __radd__(self,other): return AddFn()(self,other)
    def __sub__(self,other): return SubFn()(self,other)
    def __rsub__(self,other): return SubFn()(other,self)
    def __isub__(self,other):
        if self.do_grad and Tensor.do_grad:
            raise TypeError('in-place operation is prohibited, since it may change the graph')
        self.v=SubFn()(self,other).v
        return self

    def __pow__(self,other): return PowFn()(self,other)
    def __rpow__(self,other): return PowFn()(other,self)
    def __matmul__(self,other): return MatMulFn()(self,other)
    def __rmatmul__(self,other): return MatMulFn()(other,self)

    def __le__(self,other): return Tensor(self.v<=Tensor.strip(other))
    def __lt__(self,other): return Tensor(self.v<Tensor.strip(other))
    def __bool__(self): return bool(self.v)
    
    def __repr__(self):
        r=repr(self.v).replace('array','tensor')
        if self.fn:
            r=r[:-1]+f', fn=<{self.fn.__class__.__name__}>)'
        elif self.do_grad:
            r=r[:-1]+f', do_grad=True)'
        return r
    
    def __getitem__(self,key): return GetItemFn()(self,key)
    def __setitem__(self,key,val): self.v[key]=Tensor.strip(val)
    def __len__(self): return len(self.v)
    def __iter__(self): return iter(self.v)
    
    def cos(self): return CosFn()(self)
    def sin(self): return SinFn()(self)
    def sqrt(self): return PowFn()(self,.5)
    def sum(self,**kws): return SumFn()(self,**kws)
    def exp(self): return ExpFn()(self)
    def sigmoid(self): return 1./(1.+(-self).exp())
    def reshape(self,shape): return Tensor(self.v.reshape(shape))
    
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
            if not t.fn or create_graph: # if leaf or saving grad to every node
                # if broadcasting took place, sum-reduce grad to original shape
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

                if t.grad is None:
                    t.grad=tgrad
                else:
                    t.grad+=tgrad

            if t.fn:
                lst.extend([
                    (arg,grad)
                    for arg,grad in zip(t.fn.args,t.fn._backward(tgrad))
                    if isinstance(arg,Tensor) and arg.do_grad
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


def zeros(shape,do_grad=False):
    return Tensor(np.zeros(shape),do_grad=do_grad)
def ones(shape): return Tensor(np.ones(shape))
def randn(*args,do_grad=False):
    return Tensor(rs.randn(*args),do_grad=do_grad)
def linspace(*args): return Tensor(np.linspace(*args))
def log(t): return Tensor(np.log(Tensor.strip(t)))
def tensor(*args,**kws): return Tensor(*args,**kws)

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

class Seq(Module):
    def __init__(self,*modules):
        self.modules=modules

    def __call__(self,x):
        for m in self.modules:
            x=m(x)
        return x
