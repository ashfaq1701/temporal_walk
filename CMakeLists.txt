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

# Get Python library directory
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

# Grab Python, 3.8 or newer
find_package(pybind11 REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

pybind11_add_module(
        temporal_walk SHARED
        models/Node.cpp
        models/TemporalEdge.cpp
        models/TemporalGraph.cpp
        random/ExponentialRandomPicker.cpp
        random/LinearRandomPicker.cpp
        random/UniformRandomPicker.cpp
        TemporalWalk.cpp
        temporal_walk_wrapper.cpp
)

# Link the libraries
target_link_libraries(temporal_walk PRIVATE pybind11::module ${Python3_LIBRARIES})

target_compile_definitions(temporal_walk PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Create an executable for testing (optional)
add_executable(
        test_temporal_walk
        models/Node.cpp
        models/TemporalEdge.cpp
        models/TemporalGraph.cpp
        random/ExponentialRandomPicker.cpp
        random/LinearRandomPicker.cpp
        random/UniformRandomPicker.cpp
        TemporalWalk.cpp
        test_temporal_walk.cpp
)

# Link the executable with the temporal_walk library
target_link_libraries(test_temporal_walk PRIVATE temporal_walk)

# Set the output directory for the executable
set_target_properties(test_temporal_walk PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
