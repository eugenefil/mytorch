This is a tiny deep learning framework loosely modeled after PyTorch APIs. Made out of curiosity and for learning purposes. Like Pytorch, it supports 2 types of tensor devices:

- cpu - the default. NumPy arrays serve as tensor backend. Some routines are written in Cython for speedup.

- cuda (when libcudnn is available) - uses libcudnn for max performance. CuPy arrays serve as tensor backend to send tensors to CUDA.

- cuda (without libcudnn) - CuPy arrays are used as above, but also custom CUDA kernels to make up for libcudnn absence.
