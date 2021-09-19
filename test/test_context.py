import pytest

import optix
from optix import DeviceContext


def test_context_default():
    ctx = DeviceContext()

    assert ctx.log_callback == None
    assert not ctx.validation_mode
    assert ctx.log_callback_level == 1


def log_function(level, tag, msg):
    print("[{:>2}][{:>12}]: {}".format(level, tag, msg))


def test_context():
    ctx = DeviceContext(log_callback_function=log_function, log_callback_level=3, validation_mode=True)
    assert ctx.log_callback == log_function
    assert ctx.log_callback_level == 3
    assert ctx.validation_mode


def test_context_properties():
    ctx = DeviceContext()
    ctx.cache_enabled = True
    assert ctx.cache_enabled

    ctx.cache_enabled = False
    assert not ctx.cache_enabled

    db_sizes = (1024, 1024 * 1024)
    ctx.cache_database_sizes = db_sizes
    assert ctx.cache_database_sizes == db_sizes

    loc = ctx.cache_location
    assert type(loc) is str

    loc = "/dev/null"
    with pytest.raises(RuntimeError) as exp:
        ctx.cache_location = loc
    assert str(exp.value).startswith('OPTIX_ERROR_DISK_CACHE_INVALID_PATH')