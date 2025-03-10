import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import time
import vtk
from vtk.util import numpy_support

# Constants
GRID_SIZE = 800  # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3  # Time before a tree turns into ash
DAYS = 60  # Maximum simulation time
NUM_SIMULATIONS = multiprocessing.cpu_count()  # Number of parallel simulations

# State definitions
EMPTY = 0    # No tree
TREE = 1     # Healthy tree 
BURNING = 2  # Burning tree 
ASH = 3      # Burned tree 

def initialize_forest():
    """Creates a forest grid with all trees and ignites one random tree."""
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)  # All trees
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Tracks how long a tree burns
    
    # Ignite a random tree
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1  # Fire starts burning
    
    return forest, burn_time

def get_neighbors(x, y):
    """Returns the neighboring coordinates of a cell in the grid."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors

def save_vtk_grid(forest, filename):
    """Writes the wildfire grid data to a VTK file for visualization in ParaView."""
    grid = vtk.vtkImageData()
    grid.SetDimensions(GRID_SIZE, GRID_SIZE, 1)
    grid.SetSpacing(1.0, 1.0, 1.0)
    
    vtk_array = numpy_support.numpy_to_vtk(num_array=forest.ravel(), deep=True, array_type=vtk.VTK_INT)
    vtk_array.SetName("State")
    grid.GetPointData().AddArray(vtk_array)
    
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()

def simulate_wildfire(sim_id):
    """Simulates wildfire spread over time and saves snapshots for ParaView."""
    forest, burn_time = initialize_forest()
    fire_spread = []  # Track number of burning trees each day
    
    for day in range(DAYS):
        new_forest = forest.copy()
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1  # Increase burn time
                    
                    # If burn time exceeds threshold, turn to ash
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    
                    # Spread fire to neighbors
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1
        
        forest = new_forest.copy()
        fire_spread.append(np.sum(forest == BURNING))
        
        if np.sum(forest == BURNING) == 0:  # Stop if no more fire
            break
        
        # Save VTK file every 5 days for visualization
        if day % 5 == 0 or day == DAYS - 1:
            save_vtk_grid(forest, f"wildfire_sim_{sim_id}_day_{day}.vti")
    
    return fire_spread

# Parallel execution of simulations
def run_parallel_simulations():
    with multiprocessing.Pool(processes=NUM_SIMULATIONS) as pool:
        results = pool.map(simulate_wildfire, range(NUM_SIMULATIONS))
    
    max_days = max(len(res) for res in results)
    aggregated_results = np.zeros(max_days)
    counts = np.zeros(max_days)
    
    for res in results:
        for i, val in enumerate(res):
            aggregated_results[i] += val
            counts[i] += 1
    
    averaged_results = aggregated_results / counts
    
    return averaged_results

if __name__ == "__main__":
    # Run parallel wildfire simulations
    start_time = time.time()
    fire_spread_over_time = run_parallel_simulations()
    elapsed_time = time.time() - start_time
    print(f"Took: {elapsed_time:.4f} seconds")
    # Took 13.3897 seconds
    
    # Plot results
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(len(fire_spread_over_time)), fire_spread_over_time, label="Average Burning Trees")
    # plt.xlabel("Days")
    # plt.ylabel("Number of Burning Trees")
    # plt.title("Average Wildfire Spread Over Time (Parallel Simulations)")
    # plt.legend()
    # plt.show()
