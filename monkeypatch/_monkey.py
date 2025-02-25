import functools

from warnings import warn
WARNING = False

def patch(obj: object, name: str = '', *, backup_as=''):
    def wrapper(new):
        nonlocal name, backup_as
        name = name or new.__name__
        backup_as = backup_as or f'_{name}_'
        if hasattr(obj, name):
            old = getattr(obj, name)
            setattr(obj, backup_as, old)
            functools.wraps(old)(new)
            if WARNING: warn(f"{obj.__name__}.{name} get patched; backed up as {backup_as}")
        setattr(obj, name, new)
    return wrapper