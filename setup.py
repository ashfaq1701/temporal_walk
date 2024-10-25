from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys
import pybind11


def find_cmake():
    try:
        subprocess.check_output(['cmake', '--version'])
    except subprocess.CalledProcessError:
        raise RuntimeError("CMake must be installed to build the following extensions: _temporal_walk")


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        find_cmake()
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DPYBIND11_INCLUDE_DIR={pybind11.get_include()}'
        ]

        build_args = ['--config', 'Release']
        os.makedirs(self.build_temp, exist_ok=True)

        try:
            subprocess.check_call(['cmake', os.path.abspath('.')] + cmake_args, cwd=self.build_temp)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        except subprocess.CalledProcessError as e:
            print(f"Error during CMake configuration or build: {e}")
            sys.exit(1)


setup(
    name="temporal_walk",
    version="0.1.1",
    author="Ashfaq Salehin",
    author_email="ashfaq.salehin1701@gmail.com",
    description="A library to sample temporal walks from an in-memory temporal graph",
    long_description=open('README.md').read(),
    packages=find_packages(),
    package_data={
        'temporal_walk': ['*.so'],
    },
    include_package_data=True,
    long_description_content_type="text/markdown",
    url="https://github.com/ashfaq1701/temporal_walk",
    ext_modules=[CMakeExtension('_temporal_walk')],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=["pybind11>=2.6.0", "numpy"],
)
