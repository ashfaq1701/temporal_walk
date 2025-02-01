find_package(GTest CONFIG REQUIRED)

set(TEST_FILES
        test_temporal_walk.cpp
        test_random_picker.cpp
        test_edge_data.cpp
        test_edge_data_weights.cpp
        test_node_edge_index.cpp
        test_node_edge_index_weights.cpp
        test_node_mapping.cpp
        test_temporal_graph.cpp
        test_temporal_graph_get_node_edge_at.cpp
        test_temporal_graph_weights.cpp
        test_weight_based_random_picker.cpp
)

add_executable(test_temporal_walk
        ${TEST_FILES}
        ${SRC_FILES}
)

target_compile_features(test_temporal_walk PRIVATE cxx_std_17)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    # Set CUDA source properties for SRC_FILES only
    set_source_files_properties(
            ${CUDA_SRC_FILES}
            PROPERTIES
            LANGUAGE CUDA
    )

    # Add CUDA compile options including extended lambda
    target_compile_options(test_temporal_walk PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
            --extended-lambda
            --expt-relaxed-constexpr
            -Xcompiler=-fPIC
            >
    )

    # CUDA-specific settings
    target_compile_definitions(test_temporal_walk PRIVATE
            HAS_CUDA
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
    )

    target_link_libraries(test_temporal_walk PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
    )

    set_target_properties(test_temporal_walk PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES 89
    )
endif()

# Common libraries and includes
target_link_libraries(test_temporal_walk PRIVATE
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        GTest::gmock_main
        Boost::math
)

target_include_directories(test_temporal_walk PRIVATE
        ${CMAKE_SOURCE_DIR}/temporal_walk/src
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
