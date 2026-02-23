
import importlib.metadata
import os
import re
import subprocess
from pathlib import Path

from setuptools import find_packages, setup


def _read_env(name, default=None):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return value


def _bool_env(name, default=False):
    raw = _read_env(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _probe_cmd(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True, timeout=6).strip()
    except Exception:
        return ""


def _detect_torch_version():
    env_version = _read_env("TORCH_VERSION")
    if env_version:
        return env_version
    try:
        return importlib.metadata.version("torch")
    except Exception:
        return None


def _detect_rocm_version():
    env_version = _read_env("ROCM_VERSION")
    if env_version:
        return env_version
    out = _probe_cmd(["hipcc", "--version"])
    if not out:
        return None
    match = re.search(r"\b([0-9]+\.[0-9]+)\b", out)
    return match.group(1) if match else None


def _detect_cuda_version():
    env_version = _read_env("CUDA_VERSION")
    if env_version:
        return env_version
    out = _probe_cmd(["nvcc", "--version"])
    if not out:
        return None
    match = re.search(r"release\s+([0-9]+\.[0-9]+)", out)
    return match.group(1) if match else None


def _detect_cuda_arch_list():
    env_arch = _read_env("CUDA_ARCH_LIST")
    if env_arch:
        return env_arch

    smi = _probe_cmd(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if not smi:
        return None

    caps = []
    for line in smi.splitlines():
        cap = line.strip()
        if not cap:
            continue
        if cap.isdigit():
            cap = f"{cap}.0"
        try:
            major, minor = cap.split(".", 1)
            caps.append(f"{int(major)}.{int(minor)}")
        except Exception:
            continue

    caps = sorted(set(caps), key=lambda x: (int(x.split(".")[0]), int(x.split(".")[1])))
    return ";".join(caps) if caps else None


def _filter_supported_arches(arch_list):
    if not arch_list:
        return None
    kept = []
    for token in re.split(r"[;\s,]+", arch_list):
        token = token.strip()
        if not token:
            continue
        try:
            base = token.split("+", 1)[0]
            if float(base) >= 6.0:
                kept.append(token)
        except Exception:
            kept.append(token)
    return ";".join(kept) if kept else None


def _detect_cxx11_abi():
    raw = _read_env("CXX11_ABI")
    if raw in {"0", "1"}:
        return int(raw)
    return 1


version_vars = {}
exec("exec(open('awequant/version.py').read()); version=__version__", {}, version_vars)
awequant_version = version_vars["version"]

TORCH_VERSION = _detect_torch_version()
ROCM_VERSION = _detect_rocm_version()
CUDA_VERSION = _detect_cuda_version()
FORCE_BUILD = _bool_env("AWEQUANT_FORCE_BUILD", default=False)

build_cuda_ext = _read_env("BUILD_CUDA_EXT")
if build_cuda_ext is None:
    build_cuda_ext = "1" if (CUDA_VERSION or ROCM_VERSION) else "0"

build_awq = _bool_env("AWEQUANT_BUILD_AWQ", default=True)
build_awq_v2 = _bool_env("AWEQUANT_BUILD_AWQ_V2", default=True)

include_dirs = ["awequant_ext"]
ext_modules = []
cmdclass = {}

if build_cuda_ext == "1":
    try:
        from torch.utils import cpp_extension as cpp_ext
    except Exception as exc:
        if FORCE_BUILD:
            raise RuntimeError(
                "AWEQUANT_FORCE_BUILD=1 but torch cpp_extension is unavailable. "
                "Install torch with C++ extension support or unset AWEQUANT_FORCE_BUILD."
            ) from exc
        cpp_ext = None
        print("Warning: torch cpp_extension is unavailable, skipping CUDA extension build.")

    if cpp_ext is not None:
        max_jobs = _read_env("MAX_JOBS")
        if max_jobs:
            os.environ["MAX_JOBS"] = str(max_jobs)
            os.environ["NINJA_NUM_JOBS"] = str(max_jobs)

        arch_list = _filter_supported_arches(_detect_cuda_arch_list())
        if arch_list and not ROCM_VERSION:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

        cxx11_abi = _detect_cxx11_abi()
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17", "-DENABLE_BF16", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-DENABLE_BF16",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
            ],
        }

        if ROCM_VERSION:
            print("ROCm detected: AWQ CUDA kernels use inline PTX, skipping extension build.")
        else:
            if build_awq:
                ext_modules.append(
                    cpp_ext.CUDAExtension(
                        "awequant_awq_kernels",
                        [
                            "awequant_ext/awq/pybind_awq.cpp",
                            "awequant_ext/awq/quantization/gemm_cuda_gen.cu",
                            "awequant_ext/awq/quantization/gemv_cuda.cu",
                        ],
                        extra_compile_args=extra_compile_args,
                    )
                )

                if build_awq_v2:
                    ext_modules.append(
                        cpp_ext.CUDAExtension(
                            "awequant_awq_v2_kernels",
                            [
                                "awequant_ext/awq/pybind_awq_v2.cpp",
                                "awequant_ext/awq/quantization_new/gemv/gemv_cuda.cu",
                                "awequant_ext/awq/quantization_new/gemm/gemm_cuda.cu",
                            ],
                            extra_compile_args=extra_compile_args,
                        )
                    )

        if ext_modules:
            cmdclass["build_ext"] = cpp_ext.BuildExtension

print(f"awequant_version={awequant_version}")
print(f"BUILD_CUDA_EXT={build_cuda_ext} CUDA_VERSION={CUDA_VERSION} ROCM_VERSION={ROCM_VERSION}")
print(f"TORCH_VERSION={TORCH_VERSION}")
print(f"Extensions={[ext.name for ext in ext_modules]}")

setup(
    version=awequant_version,
    packages=find_packages(),
    include_package_data=True,
    include_dirs=include_dirs,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
