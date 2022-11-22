# Python-OptiX

Python wrapper for the OptiX 7 raytracing engine.

Python-OptiX wraps the OptiX C++ API using Cython and provides a simplified 
interface to the original C-like API using the 
[CuPy](https://cupy.dev) package.

### Supported Platforms

Only Linux is officially supported at the moment. Experimental windows support is available.

### OptiX Versions

Python-OptiX currently supports the OptiX releases 7.3.0, 7.4.0 and 7.5.0

## Installation

### Dependencies

Install a recent version of the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
and the [OptiX 7.5.0 SDK](https://developer.nvidia.com/optix/downloads/7.5.0/linux64-x86_64)

Make sure the CUDA header files are installed as well.

Add the location of CUDA to the system `PATH` variable if necessary.

Due to restrictions in OptiX's license, it is not allowed to distribute the necessary headers 
with this package in pypi. In order to enable the installation of this package in a virtual environment 
without introducing a dependency to the main system, the optix headers will be copied into the 
created wheel upon installation. 


### Using pip
```
pip install python-optix
```

### From source
```
git clone https://github.com/mortacious/python-optix.git
cd python-optix
python -m pip install .
```
