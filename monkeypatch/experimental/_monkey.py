"""Experimental"""

class ClosureProxy:
    def __init__(self, fn):
        assert callable(fn), f"{type(fn)} is not a function closure"
        assert hasattr(fn, '__code__') and hasattr(fn, '__closure__'), "closure information not available"
        assert fn.__closure__, f"No closure"
        self.__call__ = fn  # only associated with the instance
        self._freevars = fn.__code__.co_freevars
        self._closure = fn.__closure__

    @classmethod
    def try_wrap(cls, x):
        if isinstance(x, (list, tuple)):
            return type(x)(map(cls.try_wrap, x))
        try:
            return cls(x)
        except AssertionError:
            return x

    def __getattr__(self, name: str):
        if not name in self._freevars:
            raise AttributeError(f"{name} not found {self._freevars}")
        value = self._closure[self._freevars.index(name)].cell_contents
        return __class__.try_wrap(value)

    def __repr__(self):
        return repr(self.__call__) + repr(self._freevars)
