import numpy as np
import time
import statistics
from functools import wraps

def benchmark(runs=10):
    """Decorator to measure execution time of a function over multiple runs."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            for _ in range(runs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                times.append(end - start)
            
            avg_time = statistics.mean(times)
            std_dev = statistics.stdev(times)
            min_time = min(times)
            max_time = max(times)

            print(f"{func.__name__}: Avg {avg_time:.6f}s, Std {std_dev:.6f}, Min {min_time:.6f}s, Max {max_time:.6f}s")
            return result
        return wrapper
    return decorator



# DGEMM using manual nested loops (slow implementation)
@benchmark(runs=5)
def dgemm_manual(A, B, C):
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]
    return C

# DGEMM using NumPy optimized dot product (fast implementation)
@benchmark(runs=5)
def dgemm_numpy(A, B, C):
    C += np.dot(A, B)
    return C

# DGEMM using matmul
@benchmark(runs=5)
def dgemm_matmul(A, B, C):
    C += np.matmul(A, B)
    return C

matrix_sizes = [50, 100, 200]

for N in matrix_sizes:
    print(f"\nTesting DGEMM for N={N}...")

    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.zeros((N, N))

    print("Manual DGEMM:")
    C_manual = dgemm_manual(A, B, C)

    print("\nNumPy DGEMM:")
    C_numpy = dgemm_numpy(A, B, C)
    
    print("\nNumPy DGEMM matmul:")
    C_numpy = dgemm_numpy(A, B, C)
