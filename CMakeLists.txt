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
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_ARCHITECTURES 89)

        # Define that we have CUDA
        add_definitions(-DHAS_CUDA)
    else()
        message(STATUS "No CUDA compiler found. Falling back to CPU-only version.")
        set(USE_CUDA OFF)
    endif()
endif()

find_package(Boost REQUIRED COMPONENTS math)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    set(CUDA_SRC_FILES
            ${CMAKE_SOURCE_DIR}/temporal_walk/src/cuda/DualVector.cu
            ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/EdgeData.cu
            ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/NodeEdgeIndex.cu
            ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/NodeMapping.cu
            ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/TemporalGraph.cu
            ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/WeightBasedRandomPicker.cu
            ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/ExponentialIndexRandomPicker.cu
            ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/LinearRandomPicker.cu
            ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/UniformRandomPicker.cu
    )
    set(CPU_SRC_FILES
            ${CMAKE_SOURCE_DIR}/temporal_walk/src/core/TemporalWalk.cpp
    )
    set(SRC_FILES ${CUDA_SRC_FILES} ${CPU_SRC_FILES})
else()
    # Original CPU files
    set(SRC_FILES
            ${CMAKE_SOURCE_DIR}/temporal_walk/src/cuda/DualVector.cpp
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
endif()

# Include subdirectories for Python bindings, testing, and execution targets
add_subdirectory(temporal_walk/py_interface)
