import numpy as np
import pytest
from Assignment_2.dgemm.dgemm import dgemm_manual, dgemm_numpy, dgemm_matmul

def test_dgemm_correctness():
    """Test if manual DGEMM matches NumPy's matrix multiplication."""
    N = 100
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C_manual = np.zeros((N, N))
    C_numpy = np.zeros((N, N))
    C_matmul = np.zeros((N, N))

    dgemm_manual(A, B, C_manual)
    dgemm_numpy(A, B, C_numpy)
    dgemm_matmul(A, B, C_matmul)

    assert np.allclose(C_manual, C_numpy, atol=1e-6), "Manual DGEMM does not match NumPy!"
