import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
from torch.utils.hipify import hipify_python

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 6], "Requires PyTorch >= 1.6"



def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "SOFT", "kernel", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )

    hipify_ver = (
        [int(x) for x in torch.utils.hipify.__version__.split(".")]
        if hasattr(torch.utils.hipify, "__version__")
        else [0, 0, 0]
    )

    if is_rocm_pytorch and hipify_ver < [1, 0, 0]:  # TODO not needed since pt1.8

        # Earlier versions of hipification and extension modules were not
        # transparent, i.e. would require an explicit call to hipify, and the
        # hipification would introduce "hip" subdirectories, possibly changing
        # the relationship between source and header files.
        # This path is maintained for backwards compatibility.

        hipify_python.hipify(
            project_directory=this_dir,
            output_directory=this_dir,
            includes="/SOFT/kernel/csrc/*",
            show_detailed=True,
            is_pytorch_extension=True,
        )

        source_cuda = glob.glob(path.join(extensions_dir, "**", "hip", "*.hip")) + glob.glob(
            path.join(extensions_dir, "hip", "*.hip")
        )

        sources = [main_source] + sources
        sources = [
            s
            for s in sources
            if not is_rocm_pytorch or torch_ver < [1, 7] or not s.endswith("hip/vision.cpp")
        ]

    else:

        # common code between cuda and rocm platforms,
        # for hipify version [1,0,0] and later.

        source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
            path.join(extensions_dir, "*.cu")
        )

        sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        if torch_ver < [1, 7]:
            # supported by https://github.com/pytorch/pytorch/pull/43931
            CC = os.environ.get("CC", None)
            if CC is not None:
                extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "SOFT._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


# def get_model_zoo_configs() -> List[str]:
#     """
#     Return a list of configs to include in package for model zoo. Copy over these configs inside
#     detectron2/model_zoo.
#     """
#
#     # Use absolute paths while symlinking.
#     source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
#     destination = path.join(
#         path.dirname(path.realpath(__file__)), "detectron2", "model_zoo", "configs"
#     )
#     # Symlink the config directory inside package to have a cleaner pip install.
#
#     # Remove stale symlink/directory from a previous build.
#     if path.exists(source_configs_dir):
#         if path.islink(destination):
#             os.unlink(destination)
#         elif path.isdir(destination):
#             shutil.rmtree(destination)
#
#     if not path.exists(destination):
#         try:
#             os.symlink(source_configs_dir, destination)
#         except OSError:
#             # Fall back to copying if symlink fails: ex. on Windows.
#             shutil.copytree(source_configs_dir, destination)
#
#     config_paths = glob.glob("configs/**/*.yaml", recursive=True) + glob.glob(
#         "configs/**/*.py", recursive=True
#     )
#     return config_paths


setup(
    name="SOFT",
    # version=get_version(),
    author="FUDAN-zvg + HUAWEI Noah's Arks Lab + University of Surrey",
    description="SOFT:softmax-free transformer ",
    python_requires=">=3.6",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
