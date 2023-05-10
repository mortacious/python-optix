# Python-OptiX

Python-OptiX is a Python wrapper for the 
[NVIDIA OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) ray tracing engine, 
allowing for GPU-accelerated ray tracing applications in Python.

This package aims to provide a more pythonic, object-oriented interface to OptiX by wrapping the 
original C-like API using Cython. It does so by primarily relying on the [CuPy](https://cupy.dev) package.

### Supported Platforms

Only Linux is officially supported at the moment. Experimental windows support is available.

### Supported OptiX Versions

Python-OptiX always supports the most recent version of the OptiX SDK. 
The current version (1.1.0) therefore supports OptiX 7.7.0

## Installation

### Dependencies

Install a recent version of the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
as well as the [OptiX 7.7.0 SDK](https://developer.nvidia.com/optix/downloads/7.7.0/linux64-x86_64)

Make sure the CUDA header files are installed as well. 

Note, that for some variants of the CUDA Toolkit, 
like the one installed by the `conda` package manager, all required headers are not installed by default. 
`conda`-environments require installation of the additional `cudatoolkit-dev` package in 
order for this package to work.

### Environment

`python-optix` requires both the OptiX as well as the CUDA include path during setup as well as runtime 
to compile the CUDA kernels. Therefore, it is necessary to either add both locations to the system `PATH`
or set the `CUDA_PATH` and `OPTIX_PATH` variables to the respective locations.

The setup additionally has the option to embed the OptiX header files into the `python-optix` installation. 
If the variable `OPTIX_EMBED_HEADERS` is set to `1`, the setup will copy the headers from the 
OptiX SDK directory into the generated wheel.

If this option was chosen during setup, setting the `OPTIX_PATH` variable is no longer necessary as the 
embedded headers will be utilized then instead.

### Using pip
```
export OPTIX_PATH=/path/to/optix
export CUDA_PATH=/path/to/cuda_toolkit
export OPTIX_EMBED_HEADERS=1 # Optional: embedd the optix headers into the package
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

## Contributing
Contributions to Python Optix are welcome! If you find a bug or have a feature request, please open an issue on GitHub. If you want to contribute code, please fork the repository and submit a pull request.

## License
Python Optix is licensed under the MIT License. See LICENSE for details.
