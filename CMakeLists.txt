cmake_minimum_required(VERSION 3.28)
project(temporal_walk)

set(CMAKE_CXX_STANDARD 17)

# Set Python executable path
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/bin/python")

# Get pybind11 directory
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(f'{pybind11.get_include()}/../share/cmake/pybind11')"
        OUTPUT_VARIABLE PYBIND11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${PYBIND11_DIR}")
find_package(pybind11 REQUIRED)

# Automatically find all source files in src directory
file(GLOB_RECURSE SRC_FILES temporal_walk/src/*.cpp)

# Include subdirectories for Python bindings, testing, and execution targets
add_subdirectory(temporal_walk/py_interface)
add_subdirectory(temporal_walk/test)
add_subdirectory(temporal_walk/test_run)
