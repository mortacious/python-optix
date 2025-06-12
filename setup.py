from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import re
import os
from pathlib import Path
import numpy

OPTIX_COMPATIBLE_VERSION = (9, 0, 0)


# standalone import of a module (https://stackoverflow.com/a/58423785)
def import_module_from_path(path):
    """Import a module from the given path without executing any code from the hierarchy above it
    """
    import importlib
    import pathlib
    import sys

    module_path = pathlib.Path(path).resolve()
    module_name = module_path.stem  # 'path/x.py' -> 'x'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    if module not in sys.modules:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules
    return module


util = import_module_from_path('optix/path_utility.py')
cuda_include_path = util.cuda_include_path()
if cuda_include_path is None:
        raise RuntimeError("CUDA not found in the system, but is required to build this package. Consider setting"
                           "CUDA_PATH to the location of the local cuda toolkit installation.")
print("Found cuda includes at", cuda_include_path)

with util.temp_optix_include_path(OPTIX_COMPATIBLE_VERSION) as optix_include_path:
    print("Found optix includes at", optix_include_path)
    
    # get the optix version from the header for cross checking
    optix_version_re = re.compile(r'.*OPTIX_VERSION +(\d{5})')  
    with open(Path(optix_include_path) / "optix.h", 'r') as f:
        header_content = f.read()
        optix_version = int(optix_version_re.search(header_content).group(1))

    optix_version_major = optix_version // 10000
    optix_version_minor = (optix_version % 10000) // 100
    optix_version_micro = optix_version % 100

    if (optix_version_major, optix_version_minor, optix_version_micro) != OPTIX_COMPATIBLE_VERSION:
        raise ValueError(f"Found unsupported optix version {optix_version_major}.{optix_version_minor}.{optix_version_micro}. This package"
                        f"requires an optix version of {OPTIX_COMPATIBLE_VERSION[0]}.{OPTIX_COMPATIBLE_VERSION[1]}.x.")

    cython_compile_env = {
        '_OPTIX_VERSION_MAJOR': optix_version_major,
        '_OPTIX_VERSION_MINOR': optix_version_minor,
        '_OPTIX_VERSION_MICRO': optix_version_micro
    }

    libraries=[]
    if os.name == 'nt':
        # OptiX uses some Windows Registry API(e.g. RegCloseKey)
        libraries.append('advapi32')

    extensions = [Extension("*", ["optix/*.pyx"],
                            include_dirs=[str(cuda_include_path), str(optix_include_path), numpy.get_include()], libraries=libraries)]
    extensions = cythonize(extensions, language_level="3",
                            compile_time_env=cython_compile_env, build_dir="build", annotate=True)

    setup(
        ext_modules=extensions,
    )
