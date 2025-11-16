from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# build custom rasterizer
# build with `python setup.py install`
# nvcc is needed

# Ensure we target the right CUDA architectures including RTX 5090 (compute capability 8.9)
# RTX 5090 is Ada Lovelace with compute capability 8.9
nvcc_flags = [
    '-gencode=arch=compute_75,code=sm_75',   # RTX 20 series
    '-gencode=arch=compute_80,code=sm_80',   # A100
    '-gencode=arch=compute_86,code=sm_86',   # RTX 30 series
    '-gencode=arch=compute_89,code=sm_89',   # RTX 40/50 series (Ada Lovelace)
    '-gencode=arch=compute_89,code=compute_89',  # PTX for forward compatibility
    '-gencode=arch=compute_90,code=sm_90',   # H100
    '--use_fast_math',
    '-O3',
    '--ptxas-options=-v',  # Verbose PTX compilation
    '--compiler-options', '-fPIC'
]

# Add environment-based architectures if available
if 'TORCH_CUDA_ARCH_LIST' in os.environ:
    arch_list = os.environ['TORCH_CUDA_ARCH_LIST'].split(';')
    for arch in arch_list:
        if '.' in arch:
            arch_major, arch_minor = arch.split('.')
            flag = f'-gencode=arch=compute_{arch_major}{arch_minor},code=sm_{arch_major}{arch_minor}'
            if flag not in nvcc_flags:
                nvcc_flags.append(flag)

custom_rasterizer_module = CUDAExtension('custom_rasterizer_kernel', [
    'lib/custom_rasterizer_kernel/rasterizer.cpp',
    'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
    'lib/custom_rasterizer_kernel/rasterizer_gpu.cu',
], extra_compile_args={'nvcc': nvcc_flags})

setup(
    packages=find_packages(),
    version='0.1',
    name='custom_rasterizer',
    include_package_data=True,
    package_dir={'': '.'},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
