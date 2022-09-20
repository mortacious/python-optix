from struct import pack
from setuptools import setup, Extension, find_packages, find_namespace_packages
from Cython.Build import cythonize
import re
import os
from pathlib import Path
import shutil


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
print("Found cuda includes at", cuda_include_path)
print("Found optix includes at", optix_include_path)
if cuda_include_path is None or optix_include_path is None:
    raise RuntimeError("Cuda or optix not found in the system")

optix_version_re = re.compile(r'.*OPTIX_VERSION +(\d{5})')  # get the optix version from the header
with open(Path(optix_include_path) / "optix.h", 'r') as f:
    header_content = f.read()
    optix_version = int(optix_version_re.search(header_content).group(1))

optix_version_major = optix_version // 10000
optix_version_minor = (optix_version % 10000) // 100
optix_version_micro = optix_version % 100

print(f"Found OptiX version {optix_version_major}.{optix_version_minor}.{optix_version_micro}.")

cython_compile_env = {
    '_OPTIX_VERSION': optix_version,
    '_OPTIX_VERSION_MAJOR': optix_version_major,
    '_OPTIX_VERSION_MINOR': optix_version_minor,
    '_OPTIX_VERSION_MICRO': optix_version_micro
}

libraries=[]
if os.name == 'nt':
    # OptiX uses some Windows Registry API(e.g. RegCloseKey)
    libraries.append('advapi32')

extensions = [Extension("*", ["optix/*.pyx"],
                        include_dirs=[cuda_include_path, optix_include_path], libraries=libraries)]
extensions = cythonize(extensions, language_level="3",
                        compile_time_env=cython_compile_env, build_dir="build")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = import_module_from_path('optix/_version.py').__version__

package_data = {}

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    def glob_fix(package_name, glob):
        # this assumes setup.py lives in the folder that contains the package
        package_path = Path(f'./{package_name}').resolve()
        return [str(path.relative_to(package_path)) 
                for path in package_path.glob(glob)]

    class custom_bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)

            # create the path for the internal headers
            # due to optix license restrictions those headers 
            # cannot be distributed on pypi directly so we will add this headers dynamically
            # upon wheel construction to install them alongside the package

            if not os.path.exists('optix/include/optix.h'):
                shutil.copytree(optix_include_path, 'optix/include')    
            self.distribution.package_data.update({
                'optix': [*glob_fix('optix', 'include/**/*')]
            })

except ImportError:
    custom_bdist_wheel = None

setup(
    name="python-optix",
    version=version,
    author="Felix Igelbrink",
    author_email="felix.igelbrink@uni-osnabrueck.de",
    description="Python bindings to the OptiX raytracing engine by nvidia",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mortacious/python-optix",
    project_urls={
        "Bug Tracker": "https://github.com/mortacious/python-optix/issues",
    },
    packages=find_packages(exclude=['tests', 'examples']),
    ext_modules=extensions,
    install_requires=[
        'numpy',
        'cupy>=9.0'
    ],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Environment :: GPU :: NVIDIA CUDA",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    extras_require={
        'examples': ["pillow", "pyopengl", "pyglfw", "pyimgui"]
    },
    python_requires=">=3.8",
    package_data=package_data,
    zip_safe=False,
    cmdclass={'bdist_wheel': custom_bdist_wheel}
)
