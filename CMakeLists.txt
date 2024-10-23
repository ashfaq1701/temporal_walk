cmake_minimum_required(VERSION 3.28)
project(temporal_walk)

set(CMAKE_CXX_STANDARD 17)

# Get Python include directory
execute_process(
        COMMAND python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python include directory: ${PYTHON_INCLUDE_DIR}")

execute_process(
        COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
        OUTPUT_VARIABLE PYTHON_LIBRARY
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python library directory: ${PYTHON_LIBRARY}")

execute_process(
        COMMAND python3 -c "import pybind11; print(f'{pybind11.get_include()}/../share/cmake/pybind11')"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "PyBind11 include directory: ${pybind11_DIR}")

include_directories(${PYTHON_INCLUDE_DIR} ${pybind11_DIR} ${PYTHON_LIBRARY} ${PYTHON_NUMPY_INCLUDE_DIRS})

find_package(pybind11 REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

pybind11_add_module(
        _temporal_walk
        temporal_walk/_temporal_walk.cpp
        temporal_walk/core/TemporalWalk.cpp
        temporal_walk/models/TemporalEdge.cpp
        temporal_walk/models/TemporalGraph.cpp
        temporal_walk/models/Node.cpp
        temporal_walk/random/ExponentialRandomPicker.cpp
        temporal_walk/random/LinearRandomPicker.cpp
        temporal_walk/random/UniformRandomPicker.cpp
)

set_target_properties(_temporal_walk PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/temporal_walk
)

target_link_libraries(_temporal_walk PRIVATE pybind11::module ${Python3_LIBRARIES})

add_executable(
        test_temporal_walk
        temporal_walk/models/Node.cpp
        temporal_walk/models/TemporalEdge.cpp
        temporal_walk/models/TemporalGraph.cpp
        temporal_walk/random/ExponentialRandomPicker.cpp
        temporal_walk/random/LinearRandomPicker.cpp
        temporal_walk/random/UniformRandomPicker.cpp
        temporal_walk/core/TemporalWalk.cpp
        temporal_walk/test/test_temporal_walk.cpp
)

set_target_properties(test_temporal_walk PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
