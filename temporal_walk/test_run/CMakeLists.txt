add_executable(test_run_temporal_walk test_run_temporal_walk.cpp ${SRC_FILES})

set_target_properties(test_run_temporal_walk PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(test_run_temporal_walk_2 test_run_temporal_walk_2.cpp ${SRC_FILES})

set_target_properties(test_run_temporal_walk_2 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(test_run_exponential_random_picker test_run_exponential_picker.cpp ${SRC_FILES})

set_target_properties(test_run_exponential_random_picker PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

