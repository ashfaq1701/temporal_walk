#include <gtest/gtest.h>
#include "../src/data/EdgeData.cuh"

class EdgeDataTest : public ::testing::TestWithParam<bool> {
protected:
    EdgeData edges;

    EdgeDataTest(): edges(GetParam()) {}

    // Helper function to verify edge content
    void verify_edge(const size_t index, const int expected_src, const int expected_tgt, const int64_t expected_ts) const {
        ASSERT_LT(index, edges.size());
        std::visit([&](const auto& sources_vec) {
            EXPECT_EQ(sources_vec[index], expected_src);
        }, edges.sources);
        std::visit([&](const auto& targets_vec) {
            EXPECT_EQ(targets_vec[index], expected_tgt);
        }, edges.targets);
        std::visit([&](const auto& timestamps_vec) {
            EXPECT_EQ(timestamps_vec[index], expected_ts);
        }, edges.timestamps);
    }
};

// Test empty state
TEST_P(EdgeDataTest, EmptyStateTest) {
    EXPECT_TRUE(edges.empty());
    EXPECT_EQ(edges.size(), 0);
    std::visit([](const auto& vec) { EXPECT_TRUE(vec.empty()); }, edges.timestamp_group_offsets);
    std::visit([](const auto& vec) { EXPECT_TRUE(vec.empty()); }, edges.unique_timestamps);
}

// Test single edge
TEST_P(EdgeDataTest, SingleEdgeTest) {
    edges.push_back(100, 200, 100);
    EXPECT_FALSE(edges.empty());
    EXPECT_EQ(edges.size(), 1);
    verify_edge(0, 100, 200, 100);

    edges.update_timestamp_groups();
    std::visit([](const auto& vec) { EXPECT_EQ(vec.size(), 1); }, edges.unique_timestamps);
    std::visit([](const auto& vec) {
        EXPECT_EQ(vec.size(), 2);  // n+1 offsets for n groups
        EXPECT_EQ(vec[0], 0);
        EXPECT_EQ(vec[1], 1);
    }, edges.timestamp_group_offsets);
}

// Test multiple edges with same timestamp
TEST_P(EdgeDataTest, SameTimestampEdgesTest) {
    edges.push_back(100, 200, 100);
    edges.push_back(200, 300, 100);
    edges.push_back(300, 400, 100);

    edges.update_timestamp_groups();
    std::visit([](const auto& vec) { EXPECT_EQ(vec.size(), 1); }, edges.unique_timestamps);
    std::visit([](const auto& vec) {
        EXPECT_EQ(vec.size(), 2);
        EXPECT_EQ(vec[0], 0);
        EXPECT_EQ(vec[1], 3);
    }, edges.timestamp_group_offsets);
}

// Test edges with different timestamps
TEST_P(EdgeDataTest, DifferentTimestampEdgesTest) {
    edges.push_back(1, 2, 100);
    edges.push_back(2, 3, 200);
    edges.push_back(3, 4, 300);

    edges.update_timestamp_groups();
    std::visit([](const auto& vec) { EXPECT_EQ(vec.size(), 3); }, edges.unique_timestamps);
    std::visit([](const auto& vec) {
        EXPECT_EQ(vec.size(), 4);
        EXPECT_EQ(vec[0], 0);
        EXPECT_EQ(vec[1], 1);
        EXPECT_EQ(vec[2], 2);
        EXPECT_EQ(vec[3], 3);
    }, edges.timestamp_group_offsets);
}

// Test find_group functions
TEST_P(EdgeDataTest, FindGroupTest) {
    edges.push_back(100, 200, 100);
    edges.push_back(200, 300, 200);
    edges.push_back(300, 400, 300);
    edges.update_timestamp_groups();

    EXPECT_EQ(edges.find_group_after_timestamp(50), 0);
    EXPECT_EQ(edges.find_group_after_timestamp(100), 1);
    EXPECT_EQ(edges.find_group_after_timestamp(150), 1);
    EXPECT_EQ(edges.find_group_after_timestamp(200), 2);
    EXPECT_EQ(edges.find_group_after_timestamp(300), 3);
    EXPECT_EQ(edges.find_group_after_timestamp(350), 3);

    EXPECT_EQ(edges.find_group_before_timestamp(50), -1);
    EXPECT_EQ(edges.find_group_before_timestamp(150), 0);
    EXPECT_EQ(edges.find_group_before_timestamp(200), 0);
    EXPECT_EQ(edges.find_group_before_timestamp(300), 1);
    EXPECT_EQ(edges.find_group_before_timestamp(350), 2);
}

// Test timestamp group ranges
TEST_P(EdgeDataTest, TimestampGroupRangeTest) {
    edges.push_back(100, 200, 100);  // Group 0
    edges.push_back(200, 300, 100);
    edges.push_back(300, 400, 200);  // Group 1
    edges.push_back(400, 500, 300);  // Group 2
    edges.push_back(500, 600, 300);
    edges.update_timestamp_groups();

    auto [start0, end0] = edges.get_timestamp_group_range(0);
    EXPECT_EQ(start0, 0);
    EXPECT_EQ(end0, 2);

    auto [start1, end1] = edges.get_timestamp_group_range(1);
    EXPECT_EQ(start1, 2);
    EXPECT_EQ(end1, 3);

    auto [start2, end2] = edges.get_timestamp_group_range(2);
    EXPECT_EQ(start2, 3);
    EXPECT_EQ(end2, 5);

    auto [invalid_start, invalid_end] = edges.get_timestamp_group_range(3);
    EXPECT_EQ(invalid_start, 0);
    EXPECT_EQ(invalid_end, 0);
}

// Test resize
TEST_P(EdgeDataTest, ResizeTest) {
    edges.push_back(100, 200, 100);
    edges.push_back(200, 300, 200);

    edges.resize(1);
    EXPECT_EQ(edges.size(), 1);
    verify_edge(0, 100, 200, 100);

    edges.resize(3);
    EXPECT_EQ(edges.size(), 3);
    verify_edge(0, 100, 200, 100);
}

#ifdef HAS_CUDA
INSTANTIATE_TEST_SUITE_P(
    CPUAndGPU,
    EdgeDataTest,
    ::testing::Values(false, true),
    [](const testing::TestParamInfo<bool>& info) {
        return info.param ? "GPU" : "CPU";
    }
);
#else
INSTANTIATE_TEST_SUITE_P(
    CPUOnly,
    EdgeDataTest,
    ::testing::Values(false),
    [](const testing::TestParamInfo<bool>& info) {
        return "CPU";
    }
);
#endif
