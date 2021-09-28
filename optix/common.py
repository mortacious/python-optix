def round_up( val, mult_of ):
    """
    Rounds up the value to the given multiple
    """
    return val if val % mult_of == 0 else val + mult_of - val % mult_of


def ensure_iterable(obj):
    """
    Ensures that the object provided is a list or tuple and wraps it if not.
    """

    if not isinstance(obj, (list, tuple)):
        obj = (obj,)
    return obj