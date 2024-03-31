from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name="torchxx",
    ext_modules=[
        CppExtension(
            name="torchxx_cpp_ext",
            sources=[
                "src/main.cpp",
            ],
            libraries=["xxhash"],
            extra_compile_args={
                "cxx": [
                    "-g",
                    "-std=c++17",
                    "-O3",
                    "-ffast-math",
                    "-march=native",
                    # May need to remove these flags if you are not using a CPU that supports them
                    "-msse4",
                    "-mavx512cd",
                    "-mavx512er",
                    "-mavx512pf",
                    "-mavx512f",
                ],
            },
        )
    ],
    packages=find_packages(),
    py_modules=["torchxx"],
    cmdclass={"build_ext": BuildExtension},
    version="0.1.0",
)
