find_package(GTest CONFIG REQUIRED)

add_executable(test_temporal_walk test_temporal_walk.cpp ${SRC_FILES})

target_link_libraries(test_temporal_walk PRIVATE
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        GTest::gmock_main
        Boost::math
)
