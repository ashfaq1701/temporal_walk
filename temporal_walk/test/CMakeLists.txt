find_package(GTest CONFIG REQUIRED)

add_executable(
        test_temporal_walk
        test_temporal_walk.cpp
        test_edge_data.cpp
        test_node_edge_index.cpp
        test_node_mapping.cpp
        test_temporal_graph.cpp
        test_temporal_graph_get_node_edge_at.cpp
        ${SRC_FILES}
)

target_link_libraries(test_temporal_walk PRIVATE
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        GTest::gmock_main
        Boost::math
)
