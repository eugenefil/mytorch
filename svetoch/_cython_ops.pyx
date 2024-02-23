# cython: language_level=3
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

cimport cython
from cython.parallel import prange
from cython cimport floating

import numpy as np


@cython.cdivision(True) # disable python division checks (speedup)
@cython.wraparound(False) # disable python negative indexing (speedup)
@cython.boundscheck(False) # disable python bounds checking (speedup)
def im2col(floating[:, :, :, ::1] x,
           int ksize_h, int ksize_w, int stride, int padding,
           int h_out, int w_out, floating[:, :, ::1] out):
    # Above floating[:, :, :, ::1] declares a memoryview on a 4D
    # C-contiguous float or double input array:
    # - cython.floating is a fused type that accepts either float or
    #   double. Cython fused types are akin to C++ templates.
    # - ::1 in the last dimension means elements in that dimension are
    #   contiguous, i.e. contiguous C array (the default in numpy).

    cdef Py_ssize_t n, ch_in, h_in, w_in, c
    n, ch_in, h_in, w_in = x.shape[:4] # memoryview has 8 slots in shape
    c = ch_in * ksize_h * ksize_w

    cdef Py_ssize_t r, k, i, j, j_off, i_off, c_in, i_in, j_in
    cdef floating val
    # prange() parallelizes the loop with OpenMP by forking threads
    # to execute subranges (of images in our case). nogil=True is
    # required to release the Python GIL, otherwise threads wait each
    # other to acquire GIL resulting in sequential execution. nogil=True
    # also means Python objects cannot be accessed inside the loop.
    for r in prange(n, nogil=True):
        # Loop over kernel pixels from all channels and for each pixel
        # calculate its column offset and row offset from the top left
        # corner (the base) of the kernel and also its channel.
        for k in range(c):
            j_off = k % ksize_w
            i_off = k // ksize_w % ksize_h
            c_in = k // ksize_w // ksize_h

            # Now loop over all (i, j) output pixels, each of which
            # results from a kernel application. The top left corner
            # (the base) of the kernel application w.r.t. input image is:
            # i_base = i * stride - padding
            # j_base = j * stride - padding
            # To get input image position (i_in, j_in) for the k-th
            # kernel pixel in that kernel application, add kernel
            # pixel's offsets (i_off, j_off) to the kernel application's
            # base (i_base, j_base):
            # i_in = i_base + i_off = i * stride - padding + i_off
            # j_in = j_base + j_off = j * stride - padding + j_off
            for i in range(h_out):
                i_in = i * stride - padding + i_off
                for j in range(w_out):
                    j_in = j * stride - padding + j_off
                    if i_in < 0 or i_in >= h_in or j_in < 0 or j_in >= w_in:
                        val = 0. # padding
                    else:
                        val = x[r, c_in, i_in, j_in]
                    # For each output pixel with index q, extract
                    # corresponding input image pixels used for its kernel
                    # application into q-th column of the output array.
                    # There are h_out * w_out columns in total - the size
                    # of output image. There are ch_in * ksize_h * ksize_w
                    # rows - the total number of pixels in the kernel.
                    out[r, k, i * w_out + j] = val


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def col2im(floating[:, :, ::1] grad,
           int ksize_h, int ksize_w, int stride, int padding,
           int h_out, int w_out, floating[:, :, :, ::1] out):
    cdef Py_ssize_t n, ch_in, h_in, w_in, c
    n, ch_in, h_in, w_in = out.shape[:4] # memoryview has 8 slots in shape
    c = ch_in * ksize_h * ksize_w

    cdef Py_ssize_t r, k, i, j, j_off, i_off, c_in, i_in, j_in
    for r in prange(n, nogil=True):
        for k in range(c):
            j_off = k % ksize_w
            i_off = k // ksize_w % ksize_h
            c_in = k // ksize_w // ksize_h
            for i in range(h_out):
                i_in = i * stride - padding + i_off
                for j in range(w_out):
                    j_in = j * stride - padding + j_off
                    if 0 <= i_in and i_in < h_in and 0 <= j_in and j_in < w_in:
                        # When stride is less than kernel size, kernel
                        # applications overlap. This means same input
                        # pixels get used in several kernel applications
                        # and thus their gradients must be accumulated
                        # from all those applications on backward pass.
                        # The output of im2col() is the gradient here.
                        # For each gradient pixel find the corresponding
                        # input image pixel and add to its gradient.
                        out[r, c_in, i_in, j_in] += grad[r, k, i * w_out + j]


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def maxpool2d(floating[:, :, :, ::1] x,
              int ksize, int stride, int padding):
    cdef Py_ssize_t n, ch, h_in, w_in, h_out, w_out
    n, ch, h_in, w_in = x.shape[:4]
    h_out = (h_in + 2 * padding - ksize) // stride + 1
    w_out = (w_in + 2 * padding - ksize) // stride + 1

    dtype = np.float
    if floating is float: dtype = np.float32
    out = np.empty((n, ch, h_out, w_out), dtype=dtype)
    cdef floating[:, :, :, ::1] o = out

    out_idxs = np.empty((n, ch, h_out, w_out), dtype=np.intc)
    cdef int[:, :, :, ::1] idxs = out_idxs

    cdef Py_ssize_t r, k, i, j, i0, i1, j0, j1, i_in, j_in
    cdef floating maxval, v, minval = np.NINF
    cdef int maxidx
    for r in prange(n, nogil=True):
        for k in range(ch):
            for i in range(h_out):
                i0 = i * stride - padding
                i1 = i0 + ksize
                if i0 < 0: i0 = 0
                if i1 > h_in: i1 = h_in
                for j in range(w_out):
                    j0 = j * stride - padding
                    j1 = j0 + ksize
                    if j0 < 0: j0 = 0
                    if j1 > w_in: j1 = w_in
                    maxval = minval
                    maxidx =- 1
                    for i_in in range(i0, i1):
                        for j_in in range(j0, j1):
                            v = x[r, k, i_in, j_in]
                            if v > maxval:
                                maxval = v
                                maxidx = i_in * w_in + j_in
                    o[r, k, i, j] = maxval
                    idxs[r, k, i, j] = maxidx
    return out, out_idxs


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def maxpool2d_backward(floating[:, :, :, ::1] grad,
                       int[:, :, :, ::1] idxs, int h_in, int w_in):
    assert tuple(grad.shape) == tuple(idxs.shape)
    cdef Py_ssize_t n, ch, h_out, w_out
    n, ch, h_out, w_out = idxs.shape[:4]

    dtype = np.float
    if floating is float: dtype = np.float32
    out = np.zeros((n, ch, h_in, w_in), dtype=dtype)
    cdef floating[:, :, ::1] o = out.reshape(n, ch, -1)

    cdef Py_ssize_t r, k, i, j
    cdef int idx
    cdef int size = o.shape[2]
    for r in prange(n, nogil=True):
        for k in range(ch):
            for i in range(h_out):
                for j in range(w_out):
                    idx = idxs[r, k, i, j]
                    if 0 <= idx and idx < size:
                        o[r, k, idx] += grad[r, k, i, j]
    return out
