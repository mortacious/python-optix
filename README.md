# Python-OptiX

Python wrapper for the OptiX 7 raytracing engine.

Python-OptiX wraps the OptiX C++ API using Cython and provides a simplified 
interface to the original C-like API using mainly the 
[CuPy](https://cupy.dev) package.

### Supported Platforms

Only Linux is supported at the moment.

### OptiX Versions

Python-OptiX currently supports the OptiX releases 7.3.0 and 7.4.0

## Installation

### Dependencies

Install a recent version of the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
and the [OptiX 7.4.0 SDK](https://developer.nvidia.com/optix/downloads/7.4.0/linux64-x86_64)

Note: The older [OptiX 7.3.0 SDK](https://developer.nvidia.com/optix/downloads/7.4.0/linux64-x86_64) version is supported as well.

Make sure the CUDA header files are installed as well.

Add the locations of CUDA and OptiX to the system `PATH` variable if necessary.

### Using pip
```
pip install python-optix
```

### From source
```
git clone https://github.com/mortacious/python-optix.git
cd python-optix
python setup.py install
```
