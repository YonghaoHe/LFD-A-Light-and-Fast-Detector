# -*- coding: utf-8 -*-
"""
build external modules, currently including nms and focal loss
after building, the corresponding .so files are copied to the target folders
"""
import os
import shutil
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, sources, sources_cuda=[]):
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available():
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        raise EnvironmentError('external modules must be compiled with CUDA support!!')

    return extension(name=name,
                     sources=sources,
                     define_macros=define_macros,
                     extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    setup(
        name='nms',
        ext_modules=[
            make_cuda_ext(
                name='nms_ext',
                sources=['model/utils/build/nms/src/nms_ext.cpp', 'model/utils/build/nms/src/cpu/nms_cpu.cpp'],
                sources_cuda=['model/utils/build/nms/src/cuda/nms_cuda.cpp', 'model/utils/build/nms/src/cuda/nms_kernel.cu']),
            make_cuda_ext(
                name='sigmoid_focal_loss_ext',
                sources=['model/losses/build/sigmoid_focal_loss/src/sigmoid_focal_loss_ext.cpp'],
                sources_cuda=['model/losses/build/sigmoid_focal_loss/src/cuda/sigmoid_focal_loss_cuda.cu'])
        ],
        cmdclass={'build_ext': BuildExtension}
    )

    # copy .so files
    names_to_target_dirs = {'nms_ext': './model/utils/libs',
                            'sigmoid_focal_loss_ext': './model/losses/libs'}
    try:
        lib_dir = [dir_name for dir_name in os.listdir('./build') if dir_name.lower().startswith('lib.')][0]
        so_file_names = [so_file_name for so_file_name in os.listdir(os.path.join('./build', lib_dir)) if so_file_name.lower().endswith('.so')]

        for k, v in names_to_target_dirs.items():
            if not os.path.exists(v):
                os.makedirs(v)

            temp_so_file_name = [so_file_name for so_file_name in so_file_names if so_file_name.lower().startswith(k)][0]
            shutil.copyfile(os.path.join('./build', lib_dir, temp_so_file_name), os.path.join(v, temp_so_file_name))
    except:
        print('Failed to copy .so files to target folders, please check the files are generated successfully!')

    print('build and copy finished.')
