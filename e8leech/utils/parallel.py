import dask
import ray

def dask_map(func, data):
    """
    Applies a function to a list of data in parallel using Dask.
    """
    lazy_results = [dask.delayed(func)(item) for item in data]
    return dask.compute(*lazy_results)

@ray.remote
def ray_remote_function(func, item):
    """
    A remote function for Ray.
    """
    return func(item)

def ray_map(func, data):
    """
    Applies a function to a list of data in parallel using Ray.
    """
    ray.init(ignore_reinit_error=True)
    return ray.get([ray_remote_function.remote(func, item) for item in data])
