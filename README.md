# Python-OptiX

Python wrapper for the OptiX 7 raytracing engine.

Python-OptiX wraps the original OptiX C-like API using Cython while aiming to provide a more
pythonic, object-oriented interface using the [CuPy](https://cupy.dev) package.

### Supported Platforms

Only Linux is officially supported at the moment. Experimental windows support is available.

### OptiX Versions

Python-OptiX always supports the most recent version of the OptiX SDK. 
The current version therefore supports OptiX 7.6.0

## Installation

### Dependencies

Install a recent version of the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
and the [OptiX 7.6.0 SDK](https://developer.nvidia.com/optix/downloads/7.6.0/linux64-x86_64)

Make sure the CUDA header files are installed as well. 

Note, that for some variants of the CUDA Toolkit, 
like the one installed by the `conda` package manager, these are not installed by default. 
`conda`-environments require the additional `cudatoolkit-dev` package.

### Environment

`python-optix` requires both the OptiX as well as the CUDA include path during setup as well as runtime 
to compile the CUDA kernels. Therefore, it is necessary to either add both locations to the system `PATH`
or set the `CUDA_PATH` and `OPTIX_PATH` variables to the respective locations.

The setup additionally has the option to embed the OptiX header files into the `python-optix` installation. 
If the variable `OPTIX_EMBED_HEADERS` is set to `1`, the setup will copy the headers from the 
OptiX SDK directory into the generated wheel.

If this option was chosen during setup, setting the `OPTIX_PATH` is no longer required as the 
embedded headers will be utilized then.

### Using pip
```
export OPTIX_PATH=/path/to/optix
export CUDA_PATH=/path/to/cuda_toolkit
export OPTIX_EMBED_HEADERS=1 # embed the optix headers into the package
python -m pip install python-optix
```

### From source
```
git clone https://github.com/mortacious/python-optix.git
cd python-optix
export OPTIX_PATH=/path/to/optix
export CUDA_PATH=/path/to/cuda_toolkit
export OPTIX_EMBED_HEADERS=1 # embed the optix headers into the package
python -m pip install [-e] .
```
