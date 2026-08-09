import glob
import os
import platform

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
)

library_name = "uv_unwrapper"

IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_native_arch = not IS_MACOS and not IS_WINDOWS and os.getenv("USE_NATIVE_ARCH", "1") == "1"
    extension = CppExtension

    extra_link_args = []

    if IS_WINDOWS:
        # MSVC flags
        cxx_flags = ["/O2" if not debug_mode else "/Od", "/openmp"]
        if debug_mode:
            cxx_flags += ["/Z7", "/UNDEBUG"]
            extra_link_args += ["-O0"]
    elif IS_MACOS:
        cxx_flags = [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-mmacosx-version-min=11.0",
            "-arch",
            "arm64",
        ]
        # Detect the actual Apple Silicon chip family and use the matching
        # -mcpu target so the compiler can emit chip-specific instructions
        # (e.g. M4 enables SME/SME2/BF16 ISA features).
        _mcpu = "apple-m1"
        if os.getenv("USE_NATIVE_ARCH", "1") == "1":
            try:
                import subprocess as _sp
                _r = _sp.run(["sysctl", "-n", "hw.cpufamily"],
                             capture_output=True, text=True, timeout=5)
                if _r.returncode == 0 and _r.stdout.strip():
                    _fam_val = int(_r.stdout.strip())
                    _cpu_targets = {
                        0x1b588bb3: "apple-m1",
                        0xda33d83d: "apple-m2",
                        0x8765edea: "apple-m3",
                        0x6f5129ac: "apple-m4",
                        0x17d5b93a: "apple-m4",
                        0x75d4acb9: "apple-m4",
                    }
                    _mcpu = _cpu_targets.get(_fam_val, "apple-m1")
            except Exception:
                pass
        cxx_flags.append(f"-mcpu={_mcpu}")
        if debug_mode:
            cxx_flags += ["-g", "-UNDEBUG"]
            extra_link_args += ["-O0", "-g"]
    else:
        cxx_flags = [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-fopenmp",
        ]
        if use_native_arch:
            cxx_flags.append("-march=native")
        if debug_mode:
            cxx_flags += ["-g", "-UNDEBUG"]
            extra_link_args += ["-O0", "-g"]

    extra_compile_args = {"cxx": cxx_flags}

    define_macros = []
    extensions = []

    this_dir = os.path.dirname(os.path.curdir)
    sources = glob.glob(
        os.path.join(this_dir, library_name, "csrc", "**", "*.cpp"), recursive=True
    )

    if len(sources) == 0:
        print("No source files found for extension, skipping extension compilation")
        return None

    extensions.append(
        extension(
            name=f"{library_name}._C",
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=["c10", "torch", "torch_cpu", "torch_python"] if IS_MACOS else [],
        )
    )

    print(extensions)

    return extensions


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=[],
    description="Box projection based UV unwrapper",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    cmdclass={"build_ext": BuildExtension},
)
