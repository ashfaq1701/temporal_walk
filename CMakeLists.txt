cmake_minimum_required(VERSION 3.28)
project(temporal_walk)

set(CMAKE_CXX_STANDARD 17)

add_executable(
        temporal_walk
        models/Node.cpp
        models/TemporalEdge.cpp
        models/TemporalGraph.cpp
        random/ExponentialRandomPicker.cpp
        random/LinearRandomPicker.cpp
        random/UniformRandomPicker.cpp
        TemporalWalk.cpp
        test_temporal_walk.cpp
)
