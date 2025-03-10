# cythonfn.pyx
import numpy as np
cimport numpy as cnp
from cython.parallel import prange
cimport cython

cpdef cnp.ndarray[double, ndim=2] gauss_seidel(cnp.ndarray[double, ndim=2] f, int iterations=1000):
    cdef int rows = f.shape[0]
    cdef int cols = f.shape[1]
    cdef int i, j, k
    
    for k in range(iterations):
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                f[i, j] = 0.25 * (f[i, j + 1] + f[i, j - 1] +
                                  f[i + 1, j] + f[i - 1, j])
    return f
