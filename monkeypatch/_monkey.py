import functools
import contextlib

from threading import local
_threadlocal = local()
_threadlocal.fallback_stack = []

# class _Fallback:
#     """Function pointer to threadlocal fallback."""
#     def __call__(self, *args, **kwargs):
#         assert len(_threadlocal.fallback_stack), "No fallback for interception; call original function directly"
#         return _threadlocal.fallback_stack[-1](*args, **kwargs)

# fallback = _Fallback()
def get_fallback_hook():
    return _threadlocal.fallback_stack[-1]

def fallback(*args, **kwargs):
    assert len(_threadlocal.fallback_stack), "No fallback for interception; call original function directly"
    return get_fallback_hook()(*args, **kwargs)

default_backup_name = lambda origin: f"_{origin}_"

def patch(obj, name: str = ''):
    def wrapper(new):
        @contextlib.contextmanager
        def context():
            _threadlocal.fallback_stack.append(fallback)
            yield
            _threadlocal.fallback_stack.pop()

        nonlocal name
        name = name or new.__name__
        assert name, "Illegal name"
        fallback = getattr(obj, name, None)
        ctx, maybe_wrap = (
            context, functools.wraps(fallback)
        )  if fallback is not None else (
            contextlib.nullcontext, lambda f: f
        )

        @maybe_wrap
        def patched(*args, **kwargs):
            with ctx():
                return new(*args, **kwargs)
        setattr(obj, name, patched)
        return patched
    return wrapper


# The patch may change the result of __getattr__, e.g.
# ```
# def __getattr__(self, name):
#     if self.x is not None:
#         return getattr(self.x, name)
#     raise AttributeError
# ```
# If the patch modifies x from None, the fallback changes.
# If the fallback is dynamic and needed, use `intercept` instead.
def intercept(obj, name: str = '', *, times: int = 0):
    def wrapper(new):
        @contextlib.contextmanager
        def context():
            nonlocal fallback, times
            times -= 1
            setattr(obj, name, fallback) if fallback is not None else delattr(obj, name)
            yield
            if times:
                fallback = getattr(obj, name, None)  # fallback may be dynamic
                setattr(obj, name, patched)

        nonlocal name
        name = name or new.__name__
        assert name, "Illegal name"
        fallback = getattr(obj, name, None)

        def patched(*args, **kwargs):
            with context():
                return new(*args, **kwargs)
        setattr(obj, name, patched)
        return patched
    return wrapper


##############
# debug helper
##############
def _property(cls, fget=None, fset=None):
    """Property debugging helper.
    ```
    @monkey._property(A)
    def attr(self):
        return self._attr  # breakpoint here to monitor access
    @attr.setter
    def attr(self, value):
        return self._attr = value  # breakpoint here to monitor access
    ```
    """
    if not isinstance(cls, type):
        cls = type(cls)
    def getter(fget_, fset_=fset):
        assert callable(fget_)
        attr = property(fget_)
        setattr(cls, fget_.__name__, attr)
        def setter(fset__):
            assert callable(fset__)
            set_attr = attr.setter(fset__)
            setattr(cls, fget_.__name__, set_attr)
            return attr
        if fset_:
            return setter(fset_)
        # property.setter is read-only
        return type('', (property,),
                    {'setter': lambda self, fset: setter(fset)})()
    if fget:
        return getter(fget)
    return getter
