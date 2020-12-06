import concurrent.futures as cf
import numpy as np

def large_array_mean(mu : float, sigma: float):
    array = np.random.normal(mu, sigma,(10000,10000))
    return array.mean(axis = 1)

def parallel_func(*args):
    with cf.ProcessPoolExecutor() as pool:
        results = pool.map(large_array_mean, *args)

    return results