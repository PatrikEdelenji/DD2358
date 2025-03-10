import numpy as np
import time
import matplotlib.pyplot as plt
import cython
import cythonfn # type: ignore

@cython.cfunc
def gauss_seidel(f, iterations=1000):
    newf = np.copy(f)
    rows, cols = newf.shape
    
    for _ in range(iterations):
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                newf[i, j] = 0.25 * (newf[i, j + 1] + newf[i, j - 1] +
                                     newf[i + 1, j] + newf[i - 1, j])
    return newf


if __name__ == "__main__":
    grid_sizes = [10,20,50,100]
    times = []
    
    for N in grid_sizes:
        x = np.random.rand(N, N)
        
        x[0, :] = 0
        x[-1, :] = 0
        x[:, 0] = 0
        x[:, -1] = 0
        
        start_time = time.time()
        x = cythonfn.gauss_seidel(x, iterations=1000)
        elapsed_time = time.time() - start_time
        
        print(x)
        times.append(elapsed_time)
        print(f"Grid size: {N}x{N}, Time taken: {elapsed_time:.4f} seconds")
    
    plt.figure(figsize=(8, 5))
    plt.plot(grid_sizes, times, marker='o', linestyle='-')
    plt.xlabel("Grid Size (N x N)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Gauss-Seidel Solver Performance")
    plt.grid()
    plt.show()
