"""
FlashAttention CUDA Extension Setup

Build configuration for the FlashAttention PyTorch CUDA extension.

Usage:
    pip install -e .

Before building, ensure:
    - CUDA_HOME is set correctly
    - CXX points to g++ or compatible compiler
    - GPU architecture matches your hardware (see -arch flag below)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Determine CUDA architecture
# Edit this based on your GPU - common values:
# sm_61: GTX 1050/1060
# sm_70: V100
# sm_75: T4, RTX 2080
# sm_80: A100
# sm_86: RTX 3090
# sm_89: RTX 4090
CUDA_ARCH = os.environ.get('TORCH_CUDA_ARCH_LIST', 'sm_75')

setup(
    name='flash_attention',
    version='0.1.0',
    author='Your Name',
    description='Educational FlashAttention CUDA implementation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=[
        CUDAExtension(
            name='flash_attention',
            sources=['flash_attention.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    f'-arch={CUDA_ARCH}',  # GPU architecture
                    '--use_fast_math',
                    '-lineinfo',
                    '-allow-unsupported-compiler',
                    '-ccbin', 'g++',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.12.0',
    ],
)
