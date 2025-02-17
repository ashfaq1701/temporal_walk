
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

option(HAS_CUDA "Enable CUDA support" ON)

if(HAS_CUDA)
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
        set(HAS_CUDA OFF)
    endif()
endif()

set(SRC_FILES
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/WeightBasedRandomPicker.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/ExponentialIndexRandomPicker.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/LinearRandomPicker.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/random/UniformRandomPicker.cu

        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/cpu/EdgeData.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/cpu/NodeEdgeIndex.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/cpu/NodeMapping.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/cpu/TemporalGraph.cu

        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/cuda/EdgeDataCUDA.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/cuda/NodeEdgeIndexCUDA.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/cuda/NodeMappingCUDA.cu
        ${CMAKE_SOURCE_DIR}/temporal_walk/src/data/cuda/TemporalGraphCUDA.cu

        ${CMAKE_SOURCE_DIR}/temporal_walk/src/core/TemporalWalk.cu
)

if(HAS_CUDA AND CMAKE_CUDA_COMPILER)
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
    )

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -rdc=true")
    # For debug builds
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

endif()

add_subdirectory(temporal_walk/py_interface)
add_subdirectory(temporal_walk/test)
add_subdirectory(temporal_walk/test_run)
