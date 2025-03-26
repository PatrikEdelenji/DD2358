import numpy as np
import time
import matplotlib.pyplot as plt
import cython
import h5py


def gauss_seidel(f, iterations=1000):
    newf = np.copy(f)
    rows, cols = newf.shape
    
    for _ in range(iterations):
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                newf[i, j] = 0.25 * (newf[i, j + 1] + newf[i, j - 1] +
                                     newf[i + 1, j] + newf[i - 1, j])
    return newf


def save_to_hdf5(filename, dataset_name, data):
    with h5py.File(filename, 'w') as f:
        f.create_dataset(dataset_name, data=data)

if __name__ == "__main__":
    grid_sizes = [200,400,600,800,1000]
    times = []
    
    for N in grid_sizes:
        x = np.random.rand(N, N)
        
        x[0, :] = 0
        x[-1, :] = 0
        x[:, 0] = 0
        x[:, -1] = 0
        
        start_time = time.time()
        x = gauss_seidel(x, iterations=1000)
        elapsed_time = time.time() - start_time
        
        save_to_hdf5(f"gauss_seidel_output_{N}.h5", "result", x)
        
        print(x)
        times.append(elapsed_time)
        print(f"Grid size: {N}x{N}, Time taken: {elapsed_time:.4f} seconds")
        
        filename = f"gauss_seidel_output_{N}.h5"
        with h5py.File(filename, 'r') as f:
            data = f["result"][:]
            print(f"Loaded data for grid size {N}x{N}:")
            print(data)
    
    # plt.figure(figsize=(8, 5))
    # plt.plot(grid_sizes, times, marker='o', linestyle='-')
    # plt.xlabel("Grid Size (N x N)")
    # plt.ylabel("Execution Time (seconds)")
    # plt.title("Gauss-Seidel Solver Performance")
    # plt.grid()
    # plt.show()
