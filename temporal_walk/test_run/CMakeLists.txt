# Function to add an executable with common properties
function(add_cuda_executable target_name source_file)
    add_executable(${target_name}
            ${source_file}
            ${SRC_FILES}
    )

    target_compile_features(${target_name} PRIVATE cxx_std_17)

    if(USE_CUDA AND CMAKE_CUDA_COMPILER)
        # Set CUDA source properties for SRC_FILES only
        set_source_files_properties(
                ${CUDA_SRC_FILES}
                PROPERTIES
                LANGUAGE CUDA
        )

        # Add CUDA compile options including extended lambda
        target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:
                --extended-lambda
                --expt-relaxed-constexpr
                -Xcompiler=-fPIC
                >
        )

        # CUDA-specific settings
        target_compile_definitions(${target_name} PRIVATE
                HAS_CUDA
                THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
        )

        target_link_libraries(${target_name} PRIVATE
                CUDA::cudart
                CUDA::cuda_driver
        )

        set_target_properties(${target_name} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_ARCHITECTURES 89
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )
    else()
        set_target_properties(${target_name} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )
    endif()

    # Common libraries and includes
    target_link_libraries(${target_name} PRIVATE
            Boost::math
    )

    target_include_directories(${target_name} PRIVATE
            ${CMAKE_SOURCE_DIR}/temporal_walk/src
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
endfunction()

# Add executables using the function
add_cuda_executable(test_run_temporal_walk test_run_temporal_walk.cpp)
add_cuda_executable(test_run_temporal_walk_2 test_run_temporal_walk_2.cpp)
add_cuda_executable(test_run_exponential_random_picker test_run_exponential_picker.cpp)
