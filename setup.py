from setuptools import setup, Extension
from Cython.Build import cythonize


# standalone import of a module
def import_module_from_path(path):
    """Import a module from the given path."""
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


util = import_module_from_path('optix/utility.py')
cuda_include_path = util.get_cuda_include_path()
optix_include_path = util.get_optix_include_path()

extensions = [Extension("*", ["optix/*.pyx"], include_dirs=[cuda_include_path, optix_include_path])]
#extensions = [Extension("*", [s], include_dirs=[cuda_include_path, optix_include_path]) for s in source_files]


setup(
    version="0.1.0",
    author="Felix Igelbrink",
    author_email="felix.igelbrink@uni-osnabrueck.de",
    name="python-optix",
    description="Python bindings to the optix raytracing framework by nvidia",

    ext_modules=cythonize(extensions, language_level="3", annotate=True, nthreads=4),
    install_requires=[
        'numpy',
        'cupy'
    ],
    license="TBD",
    classifiers=[
        'Private :: Do Not Upload to pypi server',
    ],
)