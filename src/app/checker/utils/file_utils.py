import os
from functools import wraps
import time

def list_ext(dirpath, ext = 'txt'):
    '''
    List of file names in directory,
    which are corresponds to the extention 'ext'.

    Parameters
    ----------
    dirpath: string,
      path of directory to check.
    ext: string,
      extention of files to output.

    Returns
    -----------
    list [strings]: list of file names
    '''
    return [fname for fname in os.listdir(dirpath)
                        if fname.lower().endswith(ext)]

def log_execution(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(args)
            f,t,id = args
            print(kwargs)
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"Время выполнения для {f.name} {id}: {(end_time - start_time):.2f} секунд")
            return result
        return wrapper
    return decorator

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'Время выполнения функции {func.__name__}: {(end_time - start_time):.2f} секунд')
        return result
    return wrapper

class CustomLRUCache:
    def __init__(self, maxsize=15000):
        self.maxsize = maxsize
        self.cache = {}
        self.order = []

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Извлекаем второй аргумент и хешируем его
            key = hash(args[1])

            if key in self.cache:
                self.order.remove(key)
                self.order.append(key)
                return self.cache[key]

            if len(self.cache) >= self.maxsize:
                oldest_key = self.order.pop(0)
                del self.cache[oldest_key]

            result = func(*args, **kwargs)
            self.cache[key] = result
            self.order.append(key)
            return result
        wrapper.cache_size = lambda: len(self.cache)
        return wrapper