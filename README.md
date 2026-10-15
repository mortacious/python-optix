# Python-OptiX

[![PyPi license](https://badgen.net/pypi/license/python-optix/)](https://pypi.org/project/python-optix/) [![PyPi version](https://badgen.net/pypi/v/python-optix/)](https://pypi.org/project/python-optix/)
[![PyPI format](https://img.shields.io/pypi/format/python-optix.svg)](https://pypi.python.org/pypi/python-optix/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/python-optix.svg)](https://pypi.python.org/pypi/python-optix/)

Python-OptiX is a Python wrapper for the 
[NVIDIA OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) ray tracing engine, 
allowing for GPU-accelerated ray tracing applications in Python.

This package aims to provide a more pythonic, object-oriented interface to OptiX by wrapping the 
original C-like API using Cython. It does so by primarily relying on the [CuPy](https://cupy.dev) package.

### Supported Platforms

Only Linux is officially supported at the moment. Experimental windows support is available.

### Supported OptiX Versions

Python-OptiX always supports the most recent version of the OptiX SDK. 
The current version supports OptiX **9.1.0** (ABI 118).

**Requirements:**
- NVIDIA driver R590 or later
- CUDA Toolkit 12.0 or later
- [CuPy](https://cupy.dev)

## Installation

### Dependencies

Install a recent version of the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (12.0+)
and make sure the CUDA header files are installed as well.

Note, that for some variants of the CUDA Toolkit, 
like the one installed by the `conda` package manager, all required headers are not installed by default. 
`conda`-environments require installation of the additional `cudatoolkit-dev` package in 
order for this package to work.

The OptiX SDK itself does not need to be installed: the OptiX headers matching the supported version are
downloaded automatically from the [NVIDIA/optix-dev](https://github.com/NVIDIA/optix-dev) repository
(using `curl`), both during the build and on first use at runtime, where they are cached in
`~/.cache/python-optix/optix-dev-9.1.0`. On machines without network access, extract the
`v9.1.0` release of that repository to this location beforehand.

### Environment

`python-optix` requires the CUDA include path during setup as well as at runtime to compile the CUDA kernels. 
If CUDA is not found automatically, set the `CUDA_PATH` variable to the location of the CUDA Toolkit.

### Using pip
```
export CUDA_PATH=/path/to/cuda_toolkit # if not found automatically
python -m pip install python-optix[cupy-cuda12x]
```

### From source
```
git clone https://github.com/mortacious/python-optix.git
cd python-optix
export CUDA_PATH=/path/to/cuda_toolkit # if not found automatically
python -m pip install [-e] .
```

## Examples

The `examples` directory contains Python ports of several OptiX SDK samples
(hello, triangle, spheres, opacity micromaps, dynamic geometry and materials, the denoiser and
task-based module compilation). Their additional dependencies can be installed with
```
python -m pip install python-optix[examples]
```

## Known Issues

- With driver 595.71.05, `DeviceContext.cancel_creations` (`optixDeviceContextCancelCreations`) crashes 
  inside the driver once the context has actually compiled a module. Use `Module.cancel_creation` 
  for individual modules instead.

## Contributing
Contributions to Python Optix are welcome! If you find a bug or have a feature request, please open an issue on GitHub. If you want to contribute code, please fork the repository and submit a pull request.

## License
Python Optix is licensed under the MIT License. See LICENSE for details.
