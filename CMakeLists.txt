cmake_minimum_required(VERSION 3.28)
project(temporal_walk)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 3.10 COMPONENTS Interpreter Development REQUIRED
        PATHS ${Python3_EXECUTABLE})

execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(f'{pybind11.get_include()}/../share/cmake/pybind11')"
        OUTPUT_VARIABLE PYBIND11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${PYBIND11_DIR}")

find_package(pybind11 REQUIRED)

message(STATUS "Python include directory: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python library directory: ${Python3_LIBRARIES}")
message(STATUS "PyBind11 include directory: ${pybind11_INCLUDE_DIR}")

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
