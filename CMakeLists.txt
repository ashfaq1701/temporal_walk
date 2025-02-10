
cmake_minimum_required(VERSION 3.28)
project(temporal_walk)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/bin/python")

execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(f'{pybind11.get_include()}/../share/cmake/pybind11')"
        OUTPUT_VARIABLE PYBIND11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${PYBIND11_DIR}")
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED COMPONENTS math)

option(USE_CUDA "Enable CUDA support" ON)

if(USE_CUDA AND NOT APPLE)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_ARCHITECTURES 89)
        add_definitions(-DHAS_CUDA)
    else()
        set(USE_CUDA OFF)
    endif()
endif()

set(SRC_FILES
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/WeightBasedRandomPicker.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/ExponentialIndexRandomPicker.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/LinearRandomPicker.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/UniformRandomPicker.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/EdgeData.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/NodeEdgeIndex.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/NodeMapping.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/TemporalGraph.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/core/TemporalWalk.cu
)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
            --extended-lambda
            --expt-relaxed-constexpr
            -Xcompiler=-fPIC
    )
endif()

add_subdirectory(temporal_walk/py_interface)
add_subdirectory(temporal_walk/test)
add_subdirectory(temporal_walk/test_run)
