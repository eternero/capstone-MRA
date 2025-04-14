import time
from functools import wraps

def timing_decorator(func):
    """
    Simple decorator to measure and print the execution time of a method in seconds.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time  = time.time()
        result      = func(*args, **kwargs)
        endt_time   = time.time()
        elapse_time = endt_time - start_time

        print(f"{func.__name__} took {elapse_time:4f} seconds to complete")

        return result
    return wrapper