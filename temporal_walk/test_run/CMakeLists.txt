add_executable(test_run_temporal_walk test_run_temporal_walk.cpp ${SRC_FILES}
        print_walks.h
        ../src/models/TimestampGroupedEdges.cpp
        ../src/models/TimestampGroupedEdges.h
)

set_target_properties(test_run_temporal_walk PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(test_run_temporal_walk_2 test_run_temporal_walk_2.cpp ${SRC_FILES})

set_target_properties(test_run_temporal_walk_2 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
