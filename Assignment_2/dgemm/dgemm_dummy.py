import numpy as np

def dgemm_numpy(N):
    """
    Performs matrix multiplication using NumPy arrays.
    C = C + A * B, where A, B, and C are NxN matrices.
    """
    # Initialize matrices with random values
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.random.rand(N, N)

    # Perform matrix multiplication
    C += np.dot(A, B)

    return C  # Return the result for testing

def dgemm_numpy_fixed(A, B, C):
    """
    Performs matrix multiplication using NumPy arrays, using pre-existing A, B, C matrices.
    """
    C += np.dot(A, B)
    return C