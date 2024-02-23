import pytest

import svetoch.tensor as ten
import svetoch.nn as nn
import svetoch.nn.functional as F
import svetoch.cuda


device_params = [('cpu', False)]
if svetoch.cuda.is_available():
    device_params.extend([('cuda', False), ('cuda', True)])
@pytest.fixture(params=device_params)
def device(request):
    dev, svetoch.cuda.cudnn_enabled = request.param
    return dev


@pytest.fixture
def conv2d_inputs(device):
    x = ten.tensor([
        # image 0, channel 0
        [[[0., 1],
          [2, 3]],

        # image 0, channel 1
         [[4, 5],
          [6, 7]]],

        # image 1, channel 0
        [[[8, 9],
          [10, 11]],

        # image 1, channel 1
         [[12, 13],
          [14, 15]]]], requires_grad=True, device=device)

    weight = ten.tensor([
        # output channel 0, input channel 0
        [[[1., 1],
          [1, 1]],

        # output channel 0, input channel 1
         [[1, 1],
          [1, 1]]],

        # output channel 1, input channel 0
        [[[2, 2],
          [2, 2]],

        # output channel 1, input channel 1
         [[2, 2],
          [2, 2]]]], requires_grad=True, device=device)

    bias = ten.tensor([0., 100], requires_grad=True, device=device)
    return x, weight, bias, x.new_tensor


def test_conv2d(conv2d_inputs):
    x, weight, bias, new = conv2d_inputs
    y = F.conv2d(x, weight, bias, stride=2, padding=1)
    assert (y == new([
        [[[4., 6],
          [8, 10]],

         [[108, 112],
          [116, 120]]],

        [[[20, 22],
          [24, 26]],

         [[140, 144],
          [148, 152]]]])).all()

    y.sum().backward()
    assert (bias.grad == new([8., 8])).all()

    assert (weight.grad == new([
        [[[14., 12],
          [10, 8]],

         [[22, 20],
          [18, 16]]],

        [[[14, 12],
          [10, 8]],

         [[22, 20],
          [18, 16]]]])).all()

    assert (x.grad == new([
        [[[3., 3],
          [3, 3]],

         [[3, 3],
          [3, 3]]],

        [[[3, 3],
          [3, 3]],

         [[3, 3],
          [3, 3]]]])).all()


def test_conv2d_no_bias(conv2d_inputs):
    x, weight, _, new = conv2d_inputs
    y = F.conv2d(x, weight, stride=2, padding=1)
    assert (y == new([
        [[[4., 6],
          [8, 10]],

         [[8, 12],
          [16, 20]]],

        [[[20, 22],
          [24, 26]],

         [[40, 44],
          [48, 52]]]])).all()


def test_conv2d_unused_pixels_get_zero_grads(conv2d_inputs):
    x, _, bias, new = conv2d_inputs
    weight = new([
        [[[1.]],

         [[1]]],

        [[[2]],

         [[2]]]], requires_grad=True)

    y = F.conv2d(x, weight, bias, stride=2)
    assert (y == new([
        [[[4.]],

         [[108]]],

        [[[20]],

         [[140]]]])).all()

    y.sum().backward()
    assert (x.grad == new([
        [[[3., 0],
          [0, 0]],

         [[3, 0],
          [0, 0]]],

        [[[3, 0],
          [0, 0]],

         [[3, 0],
          [0, 0]]]])).all()


def test_conv2d_reused_pixels_accumulate_grads(conv2d_inputs):
    x, weight, bias, new = conv2d_inputs
    y = F.conv2d(x, weight, bias, padding=1)
    assert (y == new([
        [[[4., 10, 6],
          [12, 28, 16],
          [8, 18, 10]],

         [[108, 120, 112],
          [124, 156, 132],
          [116, 136, 120]]],

        [[[20, 42, 22],
          [44, 92, 48],
          [24, 50, 26]],

         [[140, 184, 144],
          [188, 284, 196],
          [148, 200, 152]]]])).all()

    y.sum().backward()
    assert (x.grad == new([
        [[[12., 12],
          [12, 12]],

         [[12, 12],
          [12, 12]]],

        [[[12, 12],
          [12, 12]],

         [[12, 12],
          [12, 12]]]])).all()


def test_relu(device):
    x = ten.tensor([
        [-1., 0],
        [3, 5]], requires_grad=True, device=device)
    y = F.relu(x)
    assert (y == x.new_tensor([
        [0., 0],
        [3, 5]])).all()

    y.sum().backward()
    assert (x.grad == x.new_tensor([
        [0., 0],
        [1, 1]])).all()


def near(x, y, eps):
    return abs(x - y) < eps


def test_kaiming_normal():
    t = ten.ones((50, 200))
    nn.init.kaiming_normal_(t)
    assert near(t.mean(), 0., .01)
    assert near(t.var(), .04, .001)

    t = ten.ones((100, 4, 5, 5))
    nn.init.kaiming_normal_(t)
    assert near(t.mean(), 0., .01)
    assert near(t.var(), .02, .001)
