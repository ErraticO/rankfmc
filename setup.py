import glob
import sys
from pathlib import Path

import numpy
from setuptools import Extension, setup

NAME = 'rankfmc'
VERSION = '0.3.1'

# define the extension packages to include
# ----------------------------------------

# prefer the generated C extensions when building
if glob.glob('rankfmc/_rankfm.c'):
    print("building extensions with pre-generated C source...")
    use_cython = False
    ext = 'c'
else:
    print("re-generating C source with cythonize...")
    from Cython.Build import cythonize
    use_cython = True
    ext = 'pyx'

# add compiler and linker arguments to optimize machine code and ignore warnings
if sys.platform == "linux":
    compile_args = ['-O2', '-ffast-math', '-fopenmp', '-Wno-unused-function', '-Wno-uninitialized']
elif sys.platform == "darwin":
    compile_args = [
        '-std=c99', 
        '-O3', 
        '-I/opt/homebrew/opt/libomp/include',
        '-L/opt/homebrew/opt/libomp/lib',
        '-lomp'
    ]
else:
    compile_args = ['/openmp', '/O2']
# define the _rankfm extension including the wrapped MT module
extensions = [
    Extension(
        name='rankfmc._rankfm',
        sources=['rankfmc/_rankfm.{ext}'.format(ext=ext), 'rankfmc/mt19937ar/mt19937ar.c'],
        extra_compile_args=compile_args
    )
]

# re-generate the C code if needed
if use_cython:
    extensions = cythonize(extensions)

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# define the main package setup function
# --------------------------------------
setup(
    name=NAME,
    version=VERSION,
    description='a python implementation of the generic factorization machines model class '
                'adapted for collaborative filtering recommendation problems '
                'with implicit feedback user-item interaction data '
                'and (optionally) additional user/item side features',
    author='ErraticO',
    author_email='wyh123132@163.com',
    url='https://github.com/ErraticO/rankfmc',
    keywords=['machine', 'learning', 'recommendation', 'factorization', 'machines', 'implicit'],
    license='GNU General Public License v3.0',
    packages=['rankfmc'],
    ext_modules=extensions,
    include_dirs=[numpy.get_include()],
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=['numpy>=1.15', 'pandas>=0.24'],
    long_description=long_description,
    long_description_content_type="text/markdown",
)

