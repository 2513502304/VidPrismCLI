from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import subprocess
import os

if __name__ == '__main__':
    ext_modules = [Extension(
        name='cy_video2txt',
        sources=[
            r'./cy_video2txt.pyx',
        ],
        include_dirs=[
            np.get_include(),
        ],
    )]

    setup(
        ext_modules=cythonize(
            ext_modules,
            annotate=True,  # 'cython -a cy_video2txt.pyx'
        ),
        script_args=[
            'build_ext',
            '-b',
            r'./build',
            '-t',
            r'./temp',
            '--inplace',
        ],
    )
