from setuptools import setup, Extension
from Cython.Build import cythonize


# standalone import of a module (https://stackoverflow.com/a/58423785)
def import_module_from_path(path):
    """Import a module from the given path without executing any code above it
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
cuda_include_path = util.get_cuda_include_path()
optix_include_path = util.get_optix_include_path()
if cuda_include_path is None or optix_include_path is None:
    raise RuntimeError("Cuda or optix not found in the system")

extensions = [Extension("*", ["optix/*.pyx"], include_dirs=[cuda_include_path, optix_include_path])]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = import_module_from_path('optix/_version.py').__version__

setup(
    name="python-optix",
    version=version,
    author="Felix Igelbrink",
    author_email="felix.igelbrink@uni-osnabrueck.de",
    description="Python bindings to the optiX raytracing framework by nvidia",
    long_description=long_description,
    long_descriptiopn_content_type="text/markdown",
    url="https://github.com/mortacious/python-optix",
    project_urls={
        "Bug Tracker": "https://github.com/mortacious/python-optix/issues",
    },
    ext_modules=cythonize(extensions, language_level="3"),
    install_requires=[
        'numpy',
        'cupy>=9.0'
    ],
    license="TBD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    extras_require={
        'examples': ["pillow"]
    },
    python_requires=">=3.8"
)
