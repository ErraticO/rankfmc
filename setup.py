"""
Minimal setup.py — only handles the Cython extension build.
All project metadata lives in pyproject.toml.
"""

import os
import subprocess
import sys

import numpy
from setuptools import Extension, setup

# ---------------------------------------------------------------------------
# Platform-specific OpenMP compiler / linker flags
# ---------------------------------------------------------------------------
if sys.platform == "linux":
    compile_args = ["-O2", "-ffast-math", "-fopenmp",
                    "-Wno-unused-function", "-Wno-uninitialized"]
    link_args = ["-fopenmp"]

elif sys.platform == "darwin":
    # Homebrew libomp provides OpenMP on macOS (clang doesn't ship it).
    try:
        libomp_prefix = subprocess.check_output(
            ["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        libomp_prefix = "/opt/homebrew/opt/libomp"

    compile_args = [
        "-O3",
        "-Xpreprocessor", "-fopenmp",
        f"-I{libomp_prefix}/include",
    ]
    link_args = [f"-L{libomp_prefix}/lib", "-lomp"]

else:  # Windows (MSVC)
    compile_args = ["/openmp", "/O2"]
    link_args = []   # MSVC links OpenMP via the compile flag only

# ---------------------------------------------------------------------------
# Cython extension — use .pyx if Cython is available, otherwise fall back
# to the pre-compiled .c file included in the sdist.
# ---------------------------------------------------------------------------
pyx_source = "rankfmc/_rankfm.pyx"
c_source    = "rankfmc/_rankfm.c"

if os.path.exists(pyx_source):
    try:
        from Cython.Build import cythonize
        sources = [pyx_source]
        extensions = cythonize(
            [
                Extension(
                    name="rankfmc._rankfm",
                    sources=sources,
                    include_dirs=[numpy.get_include()],
                    extra_compile_args=compile_args,
                    extra_link_args=link_args,
                )
            ],
            compiler_directives={"language_level": "3"},
        )
    except ImportError:
        extensions = [
            Extension(
                name="rankfmc._rankfm",
                sources=[c_source],
                include_dirs=[numpy.get_include()],
                extra_compile_args=compile_args,
                extra_link_args=link_args,
            )
        ]
else:
    # Building from sdist: .pyx not present, compile the pre-generated .c
    extensions = [
        Extension(
            name="rankfmc._rankfm",
            sources=[c_source],
            include_dirs=[numpy.get_include()],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ]

setup(ext_modules=extensions)
