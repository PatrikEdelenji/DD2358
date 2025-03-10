import numpy as np
import matplotlib.pyplot as plt
import random
import dask
import dask.array as da
from dask.delayed import delayed
from dask.distributed import Client
import time

# Constants
GRID_SIZE = 800  # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3  # Time before a tree turns into ash
DAYS = 60  # Maximum simulation time
NUM_SIMULATIONS = 8  # Number of parallel simulations

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

@delayed
def simulate_wildfire(sim_id):
    """Simulates wildfire spread over time without GUI calls."""
    forest, burn_time = initialize_forest()
    fire_spread = []  # Track number of burning trees each day
    snapshots = []  # Collect snapshots for later visualization
    
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
        
        # Store snapshot every 5 days (but do not plot here)
        if day % 5 == 0 or day == DAYS - 1:
            snapshots.append(forest.copy())
    
    return fire_spread, snapshots

def run_parallel_simulations():
    client = Client(n_workers=8, threads_per_worker=2)
    print(client.dashboard_link)
    simulations = [simulate_wildfire(i) for i in range(NUM_SIMULATIONS)]
    results = dask.compute(*simulations)
    
    # Separate fire spread data and snapshots
    fire_spread_data, snapshots_data = zip(*results)
    
    # Aggregate results (convert to Dask array for efficient processing)
    max_days = max(len(res) for res in fire_spread_data)
    aggregated_results = da.zeros(max_days)
    counts = da.zeros(max_days)
    
    for res in fire_spread_data:
        for i, val in enumerate(res):
            aggregated_results[i] += val
            counts[i] += 1
    
    averaged_results = aggregated_results / counts
    
    return averaged_results.compute(), snapshots_data

if __name__ == "__main__":
    # Run parallel wildfire simulations
    times = []
    start_time = time.time()
    fire_spread_over_time, all_snapshots = run_parallel_simulations()
    
    elapsed_time = time.time() - start_time

    times.append(elapsed_time)
    print(f"Took: {elapsed_time:.4f} seconds")
    # Took: 31.629 seconds
    
    # Plot results
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(len(fire_spread_over_time)), fire_spread_over_time, label="Average Burning Trees")
    # plt.xlabel("Days")
    # plt.ylabel("Number of Burning Trees")
    # plt.title("Average Wildfire Spread Over Time (Dask Parallel Simulations)")
    # plt.legend()
    # plt.show()
    
    # # Visualize stored snapshots in main process
    # for sim_id, snapshots in enumerate(all_snapshots):
    #     for day, forest in enumerate(snapshots):
    #         plt.figure(figsize=(6, 6))
    #         plt.imshow(forest, cmap='viridis', origin='upper')
    #         plt.title(f"Wildfire Simulation {sim_id} - Day {day*5}")
    #         plt.colorbar(label="State: 0=Empty, 1=Tree, 2=Burning, 3=Ash")
    #         plt.show()
