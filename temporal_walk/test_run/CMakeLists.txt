add_executable(test_run_temporal_walk test_run_temporal_walk.cpp ${SRC_FILES})
set_target_properties(test_run_temporal_walk PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
target_link_libraries(test_run_temporal_walk PRIVATE Boost::math)

add_executable(test_run_temporal_walk_2 test_run_temporal_walk_2.cpp ${SRC_FILES})
set_target_properties(test_run_temporal_walk_2 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
target_link_libraries(test_run_temporal_walk_2 PRIVATE Boost::math)

add_executable(test_run_exponential_random_picker test_run_exponential_picker.cpp ${SRC_FILES})
set_target_properties(test_run_exponential_random_picker PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
target_link_libraries(test_run_exponential_random_picker PRIVATE Boost::math)
