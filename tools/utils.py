import warnings


def ignore_warnings(func):
    """ Ignore warnings """

    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return inner


def iterable_wrapper(func):
    """ Wrap generator to be reusable """
    def inner(*args, **kwargs):
        return IterableWrapper(func, *args, **kwargs)
    return inner


class IterableWrapper:
    """ Wrap generator to be reusable """
    def __init__(self, func, *args, **kwargs):
        self.func, self.args, self.kwargs = func, args, kwargs
        self.generator = self.func(*self.args, **self.kwargs)

    def __iter__(self):
        self.generator = self.func(*self.args, **self.kwargs)
        return self

    def __len__(self):
        length = 0
        for _ in self.generator:
            length += 1
        return length

    def __next__(self):
        item = next(self.generator)
        if item is not None:
            return item
        else:
            raise StopIteration
