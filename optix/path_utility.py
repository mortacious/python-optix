# Code taken from the cupy installation scripts
# https://github.com/cupy/cupy/blob/master/install/build.py

# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import os
from itertools import chain

_cuda_path_cache = 'NOT_INITIALIZED'
_optix_path_cache = 'NOT_INITIALIZED'


def get_path(key):
    env = os.environ.get(key, '')
    if env:
        return env.split(os.pathsep)
    else:
        return tuple()


def search_on_path(filenames, keys=('PATH',)):
    for p in chain(*[get_path(key) for key in keys]):
        for filename in filenames:
            full = os.path.abspath(os.path.join(p, filename))
            if os.path.exists(full):
                return os.path.abspath(full)
    return None


def get_cuda_path(environment_variable='CUDA_ROOT'):
    global _cuda_path_cache

    # Use a magic word to represent the cache not filled because None is a
    # valid return value.
    if _cuda_path_cache != 'NOT_INITIALIZED':
        return _cuda_path_cache

    nvcc_path = search_on_path(('nvcc', 'nvcc.exe'), keys=(environment_variable, 'PATH'))
    cuda_path_default = None
    if nvcc_path is not None:
        cuda_path_default = os.path.normpath(
            os.path.join(os.path.dirname(nvcc_path), '..'))

    if cuda_path_default is not None:
        _cuda_path_cache = cuda_path_default
    elif os.path.exists('/usr/local/cuda'):
        _cuda_path_cache = '/usr/local/cuda'
    else:
        _cuda_path_cache = None
    return _cuda_path_cache


def get_cuda_include_path(environment_variable='CUDA_ROOT'):
    cuda_path = get_cuda_path(environment_variable=environment_variable)
    if cuda_path is None:
        return None
    cuda_include_path = os.path.join(cuda_path, "include")
    if os.path.exists(cuda_include_path):
        return cuda_include_path
    else:
        return None


def get_optix_path(environment_variable='OPTIX_PATH'):
    global _optix_path_cache

    # Use a magic word to represent the cache not filled because None is a
    # valid return value.
    if _optix_path_cache != 'NOT_INITIALIZED':
        return _optix_path_cache

    # prefer the dedicated environment variable
    optix_header_path = search_on_path(('include/optix.h',), keys=(environment_variable,))
    if optix_header_path is None:
        # search on the default path
        optix_header_path = search_on_path(('../optix/include/optix.h',), keys=('PATH',))

    if optix_header_path is not None:
        optix_header_path = os.path.normpath(os.path.join(os.path.dirname(optix_header_path), '..'))

    if optix_header_path is not None:
        _optix_path_cache = optix_header_path
    else:
        _optix_path_cache = None

    return _optix_path_cache


def get_optix_include_path(environment_variable='OPTIX_PATH'):
    optix_path = get_optix_path(environment_variable=environment_variable)
    if optix_path is None:
        return None
    optix_include_path = os.path.join(optix_path, "include")
    if os.path.exists(optix_include_path):
        return optix_include_path
    else:
        return None
