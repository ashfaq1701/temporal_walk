
function(add_test_executable target_name source_file)
    add_executable(${target_name} ${source_file} ${SRC_FILES})
    set(MOD_SRC_FILES ${SRC_FILES} ${source_file})

    target_compile_features(${target_name} PRIVATE cxx_std_17)

    if(HAS_CUDA AND CMAKE_CUDA_COMPILER)
        set_source_files_properties(
                ${MOD_SRC_FILES}
                PROPERTIES
                LANGUAGE CUDA
        )

        target_link_libraries(${target_name} PRIVATE
                CUDA::cudart
                CUDA::cuda_driver
        )

        target_include_directories(${target_name} PRIVATE
                ${CMAKE_SOURCE_DIR}/temporal_walk/src
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
    else()
        set_source_files_properties(
                ${MOD_SRC_FILES}
                PROPERTIES
                LANGUAGE CXX
        )

        target_include_directories(${target_name} PRIVATE
                ${CMAKE_SOURCE_DIR}/temporal_walk/src
        )
    endif()

    target_link_libraries(${target_name} PRIVATE Boost::math)

    set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

add_test_executable(test_run_temporal_walk test_run_temporal_walk.cpp)
add_test_executable(test_run_temporal_walk_2 test_run_temporal_walk_2.cpp)
add_test_executable(test_run_exponential_random_picker test_run_exponential_picker.cpp)
