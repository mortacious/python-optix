# Uses Code from the cupy installation scripts
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

_cuda_path = 'NOT_INITIALIZED'
_optix_path = 'NOT_INITIALIZED'


def get_path(key):
    return os.environ.get(key, '').split(os.pathsep)


def search_on_path(filenames):
    for p in get_path('PATH'):
        for filename in filenames:
            full = os.path.abspath(os.path.join(p, filename))
            if os.path.exists(full):
                return os.path.abspath(full)
    return None


def get_cuda_path():
    global _cuda_path

    # Use a magic word to represent the cache not filled because None is a
    # valid return value.
    if _cuda_path != 'NOT_INITIALIZED':
        return _cuda_path

    nvcc_path = search_on_path(('nvcc', 'nvcc.exe'))
    cuda_path_default = None
    if nvcc_path is not None:
        cuda_path_default = os.path.normpath(
            os.path.join(os.path.dirname(nvcc_path), '..'))

    #cuda_path = os.environ.get('CUDA_PATH', '')  # Nvidia default on Windows
    #if len(cuda_path) > 0 and cuda_path != cuda_path_default:
    #    print_warning(
    #        'nvcc path != CUDA_PATH',
    #        'nvcc path: %s' % cuda_path_default,
    #        'CUDA_PATH: %s' % cuda_path)

    #if os.path.exists(cuda_path):
    #    _cuda_path = cuda_path
    if cuda_path_default is not None:
        _cuda_path = cuda_path_default
    elif os.path.exists('/usr/local/cuda'):
        _cuda_path = '/usr/local/cuda'
    else:
        _cuda_path = None
    return _cuda_path


def get_cuda_include_path():
    cuda_path = get_cuda_path()
    if cuda_path is None:
        return None
    cuda_include_path = os.path.join(cuda_path, "include")
    if os.path.exists(cuda_include_path):
        return cuda_include_path
    else:
        return None


def get_optix_path():
    global _optix_path

    # Use a magic word to represent the cache not filled because None is a
    # valid return value.
    if _optix_path != 'NOT_INITIALIZED':
        return _optix_path

    optix_header_path = search_on_path(['../optix/include/optix.h'])
    if optix_header_path is not None:
        optix_header_path = os.path.normpath(os.path.join(os.path.dirname(optix_header_path), '..'))

    if optix_header_path is not None:
        _optix_path = optix_header_path
    else:
        _optix_path = None

    return _optix_path


def get_optix_include_path():
    optix_path = get_optix_path()
    if optix_path is None:
        return None
    optix_include_path = os.path.join(optix_path, "include")
    if os.path.exists(optix_include_path):
        return optix_include_path
    else:
        return None
