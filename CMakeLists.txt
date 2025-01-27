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

# Option to enable/disable CUDA
option(USE_CUDA "Enable CUDA support" ON)

# Try to find CUDA only if USE_CUDA is ON
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)

        # Set CUDA specific flags
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75)

        # Define that we have CUDA
        add_definitions(-DHAS_CUDA)
    else()
        message(STATUS "No CUDA compiler found. Falling back to CPU-only version.")
        set(USE_CUDA OFF)
    endif()
endif()

find_package(Boost REQUIRED COMPONENTS math)

set(SRC_FILES
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/EdgeData.cpp
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/NodeEdgeIndex.cpp
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/NodeMapping.cpp
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/TemporalGraph.cpp
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/ExponentialIndexRandomPicker.cpp
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/LinearRandomPicker.cpp
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/UniformRandomPicker.cpp
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/WeightBasedRandomPicker.cpp
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/core/TemporalWalk.cpp
)

# Include subdirectories for Python bindings, testing, and execution targets
add_subdirectory(temporal_walk/py_interface)
add_subdirectory(temporal_walk/test)
add_subdirectory(temporal_walk/test_run)
