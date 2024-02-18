import pytest
import torch
import torch.nn.functional as F

import svetoch

def to_torch(*ts):
    out=[torch.tensor(t.cpu().v,requires_grad=t.requires_grad,
                      device=t.device.type)
         for t in ts]
    if len(out)==1: return out[0]
    return out

@pytest.fixture(params=[('cpu',False),('cuda',False),('cuda',True)])
def conv2d_inputs(request):
    dev,cudnn_enabled=request.param
    svetoch.cudnn_enabled=cudnn_enabled
    x=svetoch.tensor([
        [[[0.,1],
          [2,3]],

         [[4,5],
          [6,7]]],

        [[[8,9],
          [10,11]],

         [[12,13],
          [14,15]]]],requires_grad=True,device=dev)

    w=svetoch.tensor([
        [[[1.,1],
          [1,1]],

         [[1,1],
          [1,1]]],

        [[[2,2],
          [2,2]],

         [[2,2],
          [2,2]]]],requires_grad=True,device=dev)

    b=svetoch.tensor([0.,100],requires_grad=True,device=dev)
    return x,w,b,x.new_tensor

def test_conv2d(conv2d_inputs):
    x,w,b,ten=conv2d_inputs
    y=svetoch.conv2d(x,w,b,stride=2,padding=1)
    assert (y==ten([
        [[[4.,6],
          [8,10]],

         [[108,112],
          [116,120]]],

        [[[20,22],
          [24,26]],

         [[140,144],
          [148,152]]]])).all()

    y.sum().backward()
    assert (b.grad==ten([8.,8])).all()

    assert (w.grad==ten([
        [[[14.,12],
          [10,8]],

         [[22,20],
          [18,16]]],

        [[[14,12],
          [10,8]],

         [[22,20],
          [18,16]]]])).all()

    assert (x.grad==ten([
        [[[3.,3],
          [3,3]],

         [[3,3],
          [3,3]]],

        [[[3,3],
          [3,3]],

         [[3,3],
          [3,3]]]])).all()

    xt,wt,bt=to_torch(x,w,b)
    yt=F.conv2d(xt,wt,bt,stride=2,padding=1)
    assert (to_torch(y)==yt).all()
    yt.sum().backward()
    assert (to_torch(b.grad)==bt.grad).all()
    assert (to_torch(w.grad)==wt.grad).all()
    assert (to_torch(x.grad)==xt.grad).all()

def test_conv2d_no_bias(conv2d_inputs):
    x,w,_,ten=conv2d_inputs
    y=svetoch.conv2d(x,w,stride=2,padding=1)
    assert (y==ten([
        [[[4.,6],
          [8,10]],

         [[8,12],
          [16,20]]],

        [[[20,22],
          [24,26]],

         [[40,44],
          [48,52]]]])).all()

def test_conv2d_unused_pixels_get_zero_grads(conv2d_inputs):
    x,_,b,ten=conv2d_inputs
    w=ten([
        [[[1.]],

         [[1]]],

        [[[2]],

         [[2]]]],requires_grad=True)

    y=svetoch.conv2d(x,w,b,stride=2)
    assert (y==ten([
        [[[4.]],

         [[108]]],

        [[[20]],

         [[140]]]])).all()

    y.sum().backward()
    assert (x.grad==ten([
        [[[3.,0],
          [0,0]],

         [[3,0],
          [0,0]]],

        [[[3,0],
          [0,0]],

         [[3,0],
          [0,0]]]])).all()

def test_conv2d_reused_pixels_accumulate_grads(conv2d_inputs):
    x,w,b,ten=conv2d_inputs
    y=svetoch.conv2d(x,w,b,padding=1)
    assert (y==ten([
        [[[4.,10,6],
          [12,28,16],
          [8,18,10]],

         [[108,120,112],
          [124,156,132],
          [116,136,120]]],

        [[[20,42,22],
          [44,92,48],
          [24,50,26]],

         [[140,184,144],
          [188,284,196],
          [148,200,152]]]])).all()

    y.sum().backward()
    assert (x.grad==ten([
        [[[12.,12],
          [12,12]],

         [[12,12],
          [12,12]]],

        [[[12,12],
          [12,12]],

         [[12,12],
          [12,12]]]])).all()

@pytest.mark.parametrize('dev,cudnn_enabled',[
    ('cpu',False),('cuda',False),('cuda',True)])
def test_relu(dev,cudnn_enabled):
    svetoch.cudnn_enabled=cudnn_enabled
    x=svetoch.tensor([
        [-1.,0],
        [3,5]],requires_grad=True,device=dev)
    y=svetoch.relu(x)
    assert (y==x.new_tensor([
        [0.,0],
        [3,5]])).all()

    y.sum().backward()
    assert (x.grad==x.new_tensor([
        [0.,0],
        [1,1]])).all()

    xt=to_torch(x)
    yt=F.relu(xt)
    assert (to_torch(y)==yt).all()
    yt.sum().backward()
    assert (to_torch(x.grad)==xt.grad).all()

def near(x,y,eps): return abs(x-y)<eps

def test_kaiming_normal():
    t=svetoch.ones((50,200))
    svetoch.kaiming_normal_(t)
    assert near(t.mean(),0.,.01)
    assert near(t.var(),.04,.001)

    t=svetoch.ones((100,4,5,5))
    svetoch.kaiming_normal_(t)
    assert near(t.mean(),0.,.01)
    assert near(t.var(),.02,.001)
