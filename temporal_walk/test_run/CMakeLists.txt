add_executable(run_temporal_walk run_temporal_walk.cpp ${SRC_FILES})

set_target_properties(run_temporal_walk PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
