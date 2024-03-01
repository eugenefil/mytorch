This is a tiny deep learning framework loosely modeled after PyTorch. It's made to try my hand at implementing automatic differentiation, CNNs, CUDA kernels, Cython, etc. It can do MNIST (like [here](https://github.com/eugenefil/ml/blob/master/mnist-svetoch.ipynb)).

It supports 2 types of tensor devices:

- `cpu` - the default. NumPy arrays serve as the tensor backend. Some routines are written in Cython and parallelized with OpenMP for speedup.

- `cuda` (when libcudnn is available) - uses libcudnn for max performance. CuPy arrays serve as the tensor backend to send tensors to GPU.

- `cuda` (without libcudnn) - CuPy arrays are used as above, but also some custom CUDA kernels to make up for libcudnn absence. You can explicitly disable libcudnn with `svetoch.cuda.cudnn_enabled = False`.

## Installation

Note, NumPy is a hard dependency, while CuPy (and thus CUDA) is optional.

```sh
git clone https://github.com/eugenefil/svetoch.git
cd svetoch
python3 -m pip install -e .
```
