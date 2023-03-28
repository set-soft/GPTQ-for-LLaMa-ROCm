from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'quant_cuda', ['quant_hip.cpp', 'quant_hip_kernel.hip']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
