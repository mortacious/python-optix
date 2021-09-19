class OptixException(RuntimeError):
    pass


def round_up( val, mult_of ):
    return val if val % mult_of == 0 else val + mult_of - val % mult_of


def ensure_iterable(obj):
    if not isinstance(obj, (list, tuple)):
        obj = (obj,)
    return obj