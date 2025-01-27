# Function to add an executable with common properties
function(add_cuda_executable target_name source_file)
    add_executable(${target_name} ${source_file} ${SRC_FILES})
    set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    target_link_libraries(${target_name} PRIVATE Boost::math)

    if(USE_CUDA AND CMAKE_CUDA_COMPILER)
        target_link_libraries(${target_name} PRIVATE CUDA::cudart)
        set_target_properties(${target_name} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
        )
    endif()
endfunction()

# Add executables using the function
add_cuda_executable(test_run_temporal_walk test_run_temporal_walk.cpp)
add_cuda_executable(test_run_temporal_walk_2 test_run_temporal_walk_2.cpp)
add_cuda_executable(test_run_exponential_random_picker test_run_exponential_picker.cpp)
