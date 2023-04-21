import typing as typ
from types import GeneratorType


def round_up(val, mult_of):
    """
    Rounds up the value to the given multiple
    """
    return val if val % mult_of == 0 else val + mult_of - val % mult_of


def ensure_iterable(obj: typ.Any) -> typ.Sequence[typ.Any]:
    """
    Ensures that the object provided is a list or tuple and wraps it if not.
    """

    if isinstance(obj, GeneratorType):
        obj = tuple(obj)  # compute the generator
    elif not isinstance(obj, (list, tuple)):
        obj = (obj,)
    return obj
