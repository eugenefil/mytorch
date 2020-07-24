import pytest
import torch
import torch.nn.functional as F

import mytorch

def to_torch(*ts):
    out=[torch.tensor(t.v,requires_grad=t.do_grad) for t in ts]
    if len(out)==1: return out[0]
    return out

@pytest.fixture
def conv2d_inputs():
    x=mytorch.tensor([
        [[[0.,1],
          [2,3]],

         [[4,5],
          [6,7]]],

        [[[8,9],
          [10,11]],

         [[12,13],
          [14,15]]]],do_grad=True)

    w=mytorch.tensor([
        [[[1.,1],
          [1,1]],

         [[1,1],
          [1,1]]],

        [[[2,2],
          [2,2]],

         [[2,2],
          [2,2]]]],do_grad=True)

    b=mytorch.tensor([0.,100],do_grad=True)
    return x,w,b

def test_conv2d(conv2d_inputs):
    x,w,b=conv2d_inputs
    y=mytorch.conv2d(x,w,b,stride=2,padding=1)
    assert (y==[
        [[[4.,6],
          [8,10]],

         [[108,112],
          [116,120]]],

        [[[20,22],
          [24,26]],

         [[140,144],
          [148,152]]]]).all()

    y.sum().backward()
    assert (b.grad==[8.,8]).all()

    assert (w.grad==[
        [[[14.,12],
          [10,8]],

         [[22,20],
          [18,16]]],

        [[[14,12],
          [10,8]],

         [[22,20],
          [18,16]]]]).all()

    assert (x.grad==[
        [[[3.,3],
          [3,3]],

         [[3,3],
          [3,3]]],

        [[[3,3],
          [3,3]],

         [[3,3],
          [3,3]]]]).all()

    xt,wt,bt=to_torch(x,w,b)
    yt=F.conv2d(xt,wt,bt,stride=2,padding=1)
    assert (to_torch(y)==yt).all()
    yt.sum().backward()
    assert (b.grad==bt.grad.numpy()).all()
    assert (w.grad==wt.grad.numpy()).all()
    assert (x.grad==xt.grad.numpy()).all()

def test_conv2d_no_bias(conv2d_inputs):
    x,w,_=conv2d_inputs
    y=mytorch.conv2d(x,w,stride=2,padding=1)
    assert (y==[
        [[[4.,6],
          [8,10]],

         [[8,12],
          [16,20]]],

        [[[20,22],
          [24,26]],

         [[40,44],
          [48,52]]]]).all()

def test_conv2d_unused_pixels_get_zero_grads(conv2d_inputs):
    x,_,b=conv2d_inputs
    w=mytorch.tensor([
        [[[1.]],

         [[1]]],

        [[[2]],

         [[2]]]],do_grad=True)

    y=mytorch.conv2d(x,w,b,stride=2)
    assert (y==[
        [[[4.]],

         [[108]]],

        [[[20]],

         [[140]]]]).all()

    y.sum().backward()
    assert (x.grad==[
        [[[3.,0],
          [0,0]],

         [[3,0],
          [0,0]]],

        [[[3,0],
          [0,0]],

         [[3,0],
          [0,0]]]]).all()

def test_conv2d_reused_pixels_accumulate_grads(conv2d_inputs):
    x,w,b=conv2d_inputs
    y=mytorch.conv2d(x,w,b,padding=1)
    assert (y==[
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
          [148,200,152]]]]).all()

    y.sum().backward()
    assert (x.grad==[
        [[[12.,12],
          [12,12]],

         [[12,12],
          [12,12]]],

        [[[12,12],
          [12,12]],

         [[12,12],
          [12,12]]]]).all()

def test_relu():
    x=mytorch.tensor([
        [-1.,0],
        [3,5]],do_grad=True)
    y=mytorch.relu(x)
    assert (y==[
        [0.,0],
        [3,5]]).all()

    y.sum().backward()
    assert (x.grad==[
        [0.,0],
        [1,1]]).all()

    xt=to_torch(x)
    yt=F.relu(xt)
    assert (to_torch(y)==yt).all()
    yt.sum().backward()
    assert (x.grad==xt.grad.numpy()).all()
