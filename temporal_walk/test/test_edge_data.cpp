#include <gtest/gtest.h>
#include "../src/data/EdgeData.h"

class EdgeDataTest : public ::testing::Test {
protected:
    EdgeData edges;

    // Helper function to verify edge content
    void verify_edge(size_t index, int expected_src, int expected_tgt, int64_t expected_ts) const {
        ASSERT_LT(index, edges.size());
        EXPECT_EQ(edges.sources[index], expected_src);
        EXPECT_EQ(edges.targets[index], expected_tgt);
        EXPECT_EQ(edges.timestamps[index], expected_ts);
    }
};

// Test empty state
TEST_F(EdgeDataTest, EmptyStateTest) {
    EXPECT_TRUE(edges.empty());
    EXPECT_EQ(edges.size(), 0);
    EXPECT_TRUE(edges.timestamp_group_offsets.empty());
    EXPECT_TRUE(edges.unique_timestamps.empty());
}

// Test single edge
TEST_F(EdgeDataTest, SingleEdgeTest) {
    edges.push_back(100, 200, 100);
    EXPECT_FALSE(edges.empty());
    EXPECT_EQ(edges.size(), 1);
    verify_edge(0, 100, 200, 100);

    edges.update_timestamp_groups();
    EXPECT_EQ(edges.unique_timestamps.size(), 1);
    EXPECT_EQ(edges.timestamp_group_offsets.size(), 2);  // n+1 offsets for n groups
    EXPECT_EQ(edges.timestamp_group_offsets[0], 0);
    EXPECT_EQ(edges.timestamp_group_offsets[1], 1);
}

// Test multiple edges with same timestamp
TEST_F(EdgeDataTest, SameTimestampEdgesTest) {
    edges.push_back(100, 200, 100);
    edges.push_back(200, 300, 100);
    edges.push_back(300, 400, 100);

    edges.update_timestamp_groups();
    EXPECT_EQ(edges.unique_timestamps.size(), 1);
    EXPECT_EQ(edges.timestamp_group_offsets.size(), 2);
    EXPECT_EQ(edges.timestamp_group_offsets[0], 0);
    EXPECT_EQ(edges.timestamp_group_offsets[1], 3);
}

// Test edges with different timestamps
TEST_F(EdgeDataTest, DifferentTimestampEdgesTest) {
    edges.push_back(1, 2, 100);
    edges.push_back(2, 3, 200);
    edges.push_back(3, 4, 300);

    edges.update_timestamp_groups();
    EXPECT_EQ(edges.unique_timestamps.size(), 3);
    EXPECT_EQ(edges.timestamp_group_offsets.size(), 4);
    EXPECT_EQ(edges.timestamp_group_offsets[0], 0);
    EXPECT_EQ(edges.timestamp_group_offsets[1], 1);
    EXPECT_EQ(edges.timestamp_group_offsets[2], 2);
    EXPECT_EQ(edges.timestamp_group_offsets[3], 3);
}

// Test find_group functions
TEST_F(EdgeDataTest, FindGroupTest) {
    edges.push_back(100, 200, 100);
    edges.push_back(200, 300, 200);
    edges.push_back(300, 400, 300);
    edges.update_timestamp_groups();

    // Test find_group_after_timestamp
    EXPECT_EQ(edges.find_group_after_timestamp(50), 0);
    EXPECT_EQ(edges.find_group_after_timestamp(100), 1);
    EXPECT_EQ(edges.find_group_after_timestamp(150), 1);
    EXPECT_EQ(edges.find_group_after_timestamp(200), 2);
    EXPECT_EQ(edges.find_group_after_timestamp(300), 3);
    EXPECT_EQ(edges.find_group_after_timestamp(350), 3);

    // Test find_group_before_timestamp
    EXPECT_EQ(edges.find_group_before_timestamp(50), -1);
    EXPECT_EQ(edges.find_group_before_timestamp(150), 0);
    EXPECT_EQ(edges.find_group_before_timestamp(200), 0);
    EXPECT_EQ(edges.find_group_before_timestamp(300), 1);
    EXPECT_EQ(edges.find_group_before_timestamp(350), 2);
}

// Test timestamp group ranges
TEST_F(EdgeDataTest, TimestampGroupRangeTest) {
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

    // Test invalid group index
    auto [invalid_start, invalid_end] = edges.get_timestamp_group_range(3);
    EXPECT_EQ(invalid_start, 0);
    EXPECT_EQ(invalid_end, 0);
}

// Test resize
TEST_F(EdgeDataTest, ResizeTest) {
    edges.push_back(100, 200, 100);
    edges.push_back(200, 300, 200);

    edges.resize(1);
    EXPECT_EQ(edges.size(), 1);
    verify_edge(0, 100, 200, 100);

    edges.resize(3);
    EXPECT_EQ(edges.size(), 3);
    verify_edge(0, 100, 200, 100);
}