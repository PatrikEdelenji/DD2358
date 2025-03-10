import cupy as cp
import time
import matplotlib.pyplot as plt

def jacobi_step(f):
    f_new = 0.25 * (cp.roll(f, shift=1, axis=0) + cp.roll(f, shift=-1, axis=0) +
                     cp.roll(f, shift=1, axis=1) + cp.roll(f, shift=-1, axis=1))
    
    f_new[0, :] = 0
    f_new[-1, :] = 0
    f_new[:, 0] = 0
    f_new[:, -1] = 0
    
    return f_new

def jacobi_solver(f, iterations=1000):
    for _ in range(iterations):
        f = jacobi_step(f)
    return f

if __name__ == "__main__":
    grid_sizes = [10, 20, 40, 100, 200, 400, 600 ,800, 1000]
    times = []
    
    for N in grid_sizes:
        
        x = cp.random.rand(N, N)
        
        x[0, :] = 0
        x[-1, :] = 0
        x[:, 0] = 0
        x[:, -1] = 0
        
        cp.cuda.Device(0).synchronize()
        start_time = time.time()
        x = jacobi_solver(x, iterations=1000)
        cp.cuda.Device(0).synchronize()
        elapsed_time = time.time() - start_time

        
        # x_cpu = cp.asnumpy(x)
        # print(x_cpu)
        
        times.append(elapsed_time)
        print(f"Grid size: {N}x{N}, Time taken: {elapsed_time:.4f} seconds")
        print(x)
    
    plt.figure(figsize=(8, 5))
    plt.plot(grid_sizes, times, marker='o', linestyle='-')
    plt.xlabel("Grid Size (N x N)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Jacobi Solver Performance on GPU (CuPy)")
    plt.grid()
    plt.show()
