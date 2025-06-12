from __future__ import annotations
import os
import shutil
from contextlib import contextmanager
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Generator, Any, Optional

_cuda_include_path = None
_optix_include_path = None


def _get_cuda_path() -> str | None:
    # Taken from cupy setup scripts
    # Use environment variable
    cuda_path = os.environ.get('CUDA_PATH', '')  # Nvidia default on Windows
    if os.path.exists(cuda_path):
        return cuda_path

    # Use nvcc path
    nvcc_path = shutil.which('nvcc')
    if nvcc_path is not None:
        return os.path.dirname(os.path.dirname(nvcc_path))

    # Use typical path
    if os.path.exists('/usr/local/cuda'):
        return '/usr/local/cuda'

    return None


def cuda_include_path() -> str | None:
    # Returns the CUDA installation path or None if not found.
    global _cuda_include_path
    if _cuda_include_path is None:
        p = _get_cuda_path()
        if p is None:
            return None
        # include the conda include directories as well
        for test_path in ("include", "targets/x86_64-linux/include"):
            include_path = os.path.join(p, test_path)
            cuda_header_path = os.path.join(include_path, "cuda.h")
            if os.path.isfile(cuda_header_path):
                _cuda_include_path = include_path
                break
    return _cuda_include_path
    

def optix_include_path(version: Optional[str | tuple] = None, path: str = "~/.cache/python-optix") -> str:
    global _optix_include_path

    if _optix_include_path is None:
        import subprocess
        import shlex

        if version is None:
            # only available in installed package
            from .context import optix_version
            version = optix_version()
        if not isinstance(version, str):
            version = ".".join((str(v) for v in version))
        path = os.path.expanduser(path)
        out_path = os.path.join(path, f"optix-dev-{version}")

        if not os.path.exists(out_path):
            os.makedirs(path, exist_ok=True)

            out_file_dl = out_path + ".tar.gz"
            command = f"curl -LJ -o {out_file_dl} https://github.com/NVIDIA/optix-dev/archive/refs/tags/v{version}.tar.gz"
            args = shlex.split(command)
            #print("downloading optix headers with command", command)
            try:
                subprocess.check_call(args, shell=False)
            except subprocess.CalledProcessError as e:
                raise ValueError(f"Unable to download optix headers for version {version} into {path}.") from e

            command = f"tar -xf {out_file_dl} -C {path}"
            args = shlex.split(command)
            #print("extracting optix headers with command", args)

            try:
                subprocess.check_call(args, shell=False)
            except subprocess.CalledProcessError as e:
                raise ValueError(f"Unable to extract optix headers at path {out_file_dl}.") from e

        optix_h_file = os.path.join(out_path, "include/optix.h")
        if not os.path.exists(optix_h_file):
            raise ValueError(f"Result path {out_path} does not contain the optix headers.")
        
        _optix_include_path = os.path.join(out_path, "include")

    return _optix_include_path


@contextmanager
def temp_optix_include_path(version: str | tuple) -> Generator[Any, Any, Any]:
    temp_dir = os.path.abspath(".tmp")
    os.makedirs(temp_dir, exist_ok=True)
    #with tempfile.TemporaryDirectory() as td:
    optix_path = optix_include_path(version, path=temp_dir)
    yield optix_path
    shutil.rmtree(temp_dir)



