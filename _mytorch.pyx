# cython: language_level=3
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

cimport cython
from cython.parallel import prange
from cython cimport floating

import numpy as np

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def extract_kernels(floating[:,:,:,::1] x,
                    int ksize_h,int ksize_w,
                    int stride,int padding):
    cdef Py_ssize_t n,ch_in,h_in,w_in,c,h_out,w_out
    n,ch_in,h_in,w_in=x.shape[:4] # memoryview has 8 slots in shape
    c=ch_in*ksize_h*ksize_w
    h_out=(h_in+2*padding-ksize_h)//stride+1
    w_out=(w_in+2*padding-ksize_w)//stride+1

    dtype=np.float # default float is usually float64
    if floating is float: dtype=np.float32
    out=np.empty((n,c,h_out*w_out),dtype=dtype)
    cdef floating[:,:,::1] o=out

    cdef Py_ssize_t r,k,i,j,w_off,h_off,c_in,i_in,j_in
    cdef floating val
    for r in prange(n,nogil=True):
        for k in range(c):
            w_off=k%ksize_w
            h_off=k//ksize_w%ksize_h
            c_in=k//ksize_w//ksize_h
            for i in range(h_out):
                i_in=i*stride+h_off-padding
                for j in range(w_out):
                    j_in=j*stride+w_off-padding
                    if i_in<0 or i_in>=h_in or j_in<0 or j_in>=w_in:
                        val=0. # padding
                    else:
                        val=x[r,c_in,i_in,j_in]
                    o[r,k,i*w_out+j]=val
    return out

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def extract_kernels_backward(floating[:,:,::1] grad,
                             int ch_in,int h_in,int w_in,
                             int ksize_h,int ksize_w,
                             int stride,int padding):
    cdef Py_ssize_t n,c,h_out_by_w_out,h_out,w_out
    n,c,h_out_by_w_out=grad.shape[:3] # memoryview has 8 slots in shape
    assert c == ch_in*ksize_h*ksize_w

    h_out=(h_in+2*padding-ksize_h)//stride+1
    w_out=(w_in+2*padding-ksize_w)//stride+1
    assert h_out_by_w_out == h_out*w_out

    dtype=np.float # default float is usually float64
    if floating is float: dtype=np.float32
    out=np.zeros((n,ch_in,h_in,w_in),dtype=dtype)
    cdef floating[:,:,:,::1] o=out

    cdef Py_ssize_t r,k,i,j,w_off,h_off,c_in,i_in,j_in
    for r in prange(n,nogil=True):
        for k in range(c):
            w_off=k%ksize_w
            h_off=k//ksize_w%ksize_h
            c_in=k//ksize_w//ksize_h
            for i in range(h_out):
                i_in=i*stride+h_off-padding
                for j in range(w_out):
                    j_in=j*stride+w_off-padding
                    if 0<=i_in and i_in<h_in and 0<=j_in and j_in<w_in:
                        o[r,c_in,i_in,j_in]+=grad[r,k,i*w_out+j]
    return out

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def maxpool2d(floating[:,:,:,::1] x,
              int ksize,int stride,int padding):
    cdef Py_ssize_t n,ch,h_in,w_in,h_out,w_out
    n,ch,h_in,w_in=x.shape[:4]
    h_out=(h_in+2*padding-ksize)//stride+1
    w_out=(w_in+2*padding-ksize)//stride+1

    dtype=np.float
    if floating is float: dtype=np.float32
    out=np.empty((n,ch,h_out,w_out),dtype=dtype)
    cdef floating[:,:,:,::1] o=out

    out_idxs=np.empty((n,ch,h_out,w_out),dtype=np.intc)
    cdef int[:,:,:,::1] idxs=out_idxs

    cdef Py_ssize_t r,k,i,j,i0,i1,j0,j1,i_in,j_in
    cdef floating maxval,v,minval=np.NINF
    cdef int maxidx
    for r in prange(n,nogil=True):
        for k in range(ch):
            for i in range(h_out):
                i0=i*stride-padding
                i1=i0+ksize
                if i0<0: i0=0
                if i1>h_in: i1=h_in
                for j in range(w_out):
                    j0=j*stride-padding
                    j1=j0+ksize
                    if j0<0: j0=0
                    if j1>w_in: j1=w_in
                    maxval=minval
                    maxidx=-1
                    for i_in in range(i0,i1):
                        for j_in in range(j0,j1):
                            v=x[r,k,i_in,j_in]
                            if v>maxval:
                                maxval=v
                                maxidx=i_in*w_in+j_in
                    o[r,k,i,j]=maxval
                    idxs[r,k,i,j]=maxidx
    return out,out_idxs

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def maxpool2d_backward(floating[:,:,:,::1] grad,
                       int[:,:,:,::1] idxs,int h_in,int w_in):
    assert tuple(grad.shape)==tuple(idxs.shape)
    cdef Py_ssize_t n,ch,h_out,w_out
    n,ch,h_out,w_out=idxs.shape[:4]

    dtype=np.float
    if floating is float: dtype=np.float32
    out=np.zeros((n,ch,h_in,w_in),dtype=dtype)
    cdef floating[:,:,::1] o=out.reshape(n,ch,-1)

    cdef Py_ssize_t r,k,i,j
    cdef int idx
    cdef int size=o.shape[2]
    for r in prange(n,nogil=True):
        for k in range(ch):
            for i in range(h_out):
                for j in range(w_out):
                    idx=idxs[r,k,i,j]
                    if 0<=idx and idx<size:
                        o[r,k,idx]+=grad[r,k,i,j]
    return out
