import timeit

class Timer:
    def __init__(self, name):
        self.name = name
        print(self.name + '...')

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        print(f'{self.name} took {timeit.default_timer() - self.start:.2f}s')
