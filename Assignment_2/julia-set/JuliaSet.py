"""Julia set generator without optional PIL-based image drawing"""
import time
import timeit
from functools import wraps
import numpy as np
from timeit import default_timer as timer
import statistics


# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -.42193

timing_data = {}


#def run_benchmark():
#    import JuliaSet
#    f = lambda: JuliaSet.calc_pure_python(1000, 300)
#    
    # Collect timing results
#    times = timeit.repeat(f, number=1, repeat=10)

    # Compute average and standard deviation
#    avg_time = statistics.mean(times)
#    std_dev = statistics.stdev(times)

    # Print with higher precision
#    print(f"Average Time: {avg_time:.9f} seconds")
#    print(f"Standard Deviation: {std_dev:.9f} seconds")
    
# decorator to time
def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
        return result
    return measure_time

def checktick(timer_function):
    """
    Measure the clock granularity of a given timer function.
    Returns the minimum measurable time difference in seconds.
    """
    M = 200  # Number of measurements
    timesfound = np.empty((M,))
    for i in range(M):
        t1 = timer_function()
        t2 = timer_function()
        while (t2 - t1) <= 0:
            t2 = timer_function()
        timesfound[i] = t1
    minDelta = np.diff(timesfound).min()
    return minDelta


#@profile
def calc_pure_python(desired_width, max_iterations):
    """Create a list of complex coordinates (zs) and complex parameters (cs),
    build Julia set"""
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    # build a list of coordinates and the initial condition for each cell.
    # Note that our initial condition is a constant and could easily be removed,
    # we use it to simulate a real-world scenario with several inputs to our
    # function
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    print("Length of x:", len(x))
    print("Total elements:", len(zs))
    start_time = time.time()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time = time.time()
    secs = end_time - start_time
    print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")

    # This sum is expected for a 1000^2 grid with 300 iterations
    # It ensures that our code evolves exactly as we'd intended
    #assert sum(output) == 33219980
    return output
    

#@profile
def calculate_z_serial_purepython(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
    #with profile.timestamp("calculate_output_list"):
    output = [0] * len(zs)
    time.sleep(1)
    #with profile.timestamp("calculate_output"):
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z * z + c
            n += 1
        output[i] = n
    return output




if __name__ == "__main__":
    # Calculate the Julia set using a pure Python solution with
    # reasonable defaults for a laptop
    start_time1 = time.time()
    calc_pure_python(desired_width=1000, max_iterations=300) #lowered to 100 for memory_profiler
    end_time1 = time.time()
    secs = end_time1 - start_time1
    print("Whole process took", secs, "seconds")
    # run_benchmark()
    
    # granularity_time = checktick(time.time)
    # granularity_timer = checktick(timer)
    # granularity_time_ns = checktick(lambda: time.time_ns() * 1e-9)  # Convert nanoseconds to seconds

    # print(f"Granularity of time.time(): {granularity_time:.10f} seconds")
    # print(f"Granularity of timeit.default_timer(): {granularity_timer:.10f} seconds")
    # print(f"Granularity of time.time_ns(): {granularity_time_ns:.10f} seconds")
    
    
