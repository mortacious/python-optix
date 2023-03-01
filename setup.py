from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import re
import os
from pathlib import Path
import shutil
import numpy

OPTIX_COMPATIBLE_VERSION = (7, 6)

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
cuda_include_path = util.get_cuda_include_path(environment_variable='CUDA_PATH')
optix_include_path = util.get_optix_include_path(environment_variable='OPTIX_PATH')
print("Found cuda includes at", cuda_include_path)
print("Found optix includes at", optix_include_path)
if cuda_include_path is None:
    raise RuntimeError("CUDA not found in the system, but is required to build this package. Consider setting"
                       "CUDA_PATH to the location of the local cuda toolkit installation.")
if optix_include_path is None:
    raise RuntimeError("OptiX not found in the system, but is required to build this package. Consider setting "
                       "OPTIX_PATH to the location of the optix SDK.")

optix_version_re = re.compile(r'.*OPTIX_VERSION +(\d{5})')  # get the optix version from the header
with open(Path(optix_include_path) / "optix.h", 'r') as f:
    header_content = f.read()
    optix_version = int(optix_version_re.search(header_content).group(1))

optix_version_major = optix_version // 10000
optix_version_minor = (optix_version % 10000) // 100
optix_version_micro = optix_version % 100

if (optix_version_major, optix_version_minor) != OPTIX_COMPATIBLE_VERSION:
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
                        include_dirs=[cuda_include_path, optix_include_path, numpy.get_include()], libraries=libraries)]
extensions = cythonize(extensions, language_level="3",
                        compile_time_env=cython_compile_env, build_dir="build", annotate=True)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = import_module_from_path('optix/_version.py').__version__

package_data = {}


def glob_fix(package_name, glob):
    # this assumes setup.py lives in the folder that contains the package
    package_path = Path(f'./{package_name}').resolve()
    return [str(path.relative_to(package_path))
            for path in package_path.glob(glob)]

from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop

class EmbeddHeadersCommandMixin:
    def update_package_data(self):
        self.distribution.package_data.update({
            'optix': [*glob_fix('optix', 'include/**/*')]
        })
        print("embedding optix headers into package data",
              self.distribution.package_data)

    def run(self):
        embedd = os.getenv("OPTIX_EMBED_HEADERS")
        if embedd:
            # create the path for the internal headers
            # due to optix license restrictions those headers
            # cannot be distributed on pypi directly so we will add this headers dynamically
            # upon wheel construction to install them alongside the package

            if not os.path.exists('optix/include/optix.h'):
                shutil.copytree(optix_include_path, 'optix/include')

            self.update_package_data()

        super().run()


class CustomInstallCommand(EmbeddHeadersCommandMixin, _install):
    pass


class CustomDevelopCommand(EmbeddHeadersCommandMixin, _develop):
    pass


cmd_classes = {'install': CustomInstallCommand,
               'develop': CustomDevelopCommand}

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class CustomBdistWheelCommand(EmbeddHeadersCommandMixin, _bdist_wheel):
        pass
    cmd_classes['bdist_wheel'] = CustomBdistWheelCommand
except ImportError:
    CustomBdistWheel = None

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
    cmdclass=cmd_classes
)
