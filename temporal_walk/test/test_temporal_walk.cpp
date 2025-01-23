#include <gtest/gtest.h>
#include <cmath>

#include "test_utils.h"
#include "../src/core/TemporalWalk.h"

constexpr int TEST_NODE_ID = 42;
constexpr int MAX_WALK_LEN = 20;
constexpr int64_t MAX_TIME_CAPACITY = 5;

constexpr RandomPickerType exponential_picker_type = RandomPickerType::ExponentialIndex;
constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;

class EmptyTemporalWalkTest : public ::testing::Test {
protected:
    void SetUp() override {
        temporal_walk = std::make_unique<TemporalWalk>(true, -1, true, -1);
    }

    std::unique_ptr<TemporalWalk> temporal_walk;
};

class EmptyTemporalWalkTestWithMaxCapacity : public ::testing::Test {
protected:
    void SetUp() override {
        temporal_walk = std::make_unique<TemporalWalk>(true, MAX_TIME_CAPACITY, true, -1);
    }

    std::unique_ptr<TemporalWalk> temporal_walk;
};

class FilledDirectedTemporalWalkTest : public ::testing::Test {
protected:
    FilledDirectedTemporalWalkTest() {
        sample_edges = read_edges_from_csv("../../../data/sample_data.csv");
    }

    void SetUp() override {
        temporal_walk = std::make_unique<TemporalWalk>(true, -1, true, -1);
        temporal_walk->add_multiple_edges(sample_edges);
    }

    std::vector<std::tuple<int, int, int64_t>> sample_edges;
    std::unique_ptr<TemporalWalk> temporal_walk;
};

class FilledUndirectedTemporalWalkTest : public ::testing::Test {
protected:
    FilledUndirectedTemporalWalkTest() {
        sample_edges = read_edges_from_csv("../../../data/sample_data.csv");
    }

    void SetUp() override {
        temporal_walk = std::make_unique<TemporalWalk>(false, -1, true, -1);
        temporal_walk->add_multiple_edges(sample_edges);
    }

    std::vector<std::tuple<int, int, int64_t>> sample_edges;
    std::unique_ptr<TemporalWalk> temporal_walk;
};

class TimescaleBoundedTemporalWalkTest : public ::testing::Test {
protected:
    void SetUp() override {
        temporal_walk = std::make_unique<TemporalWalk>(true, -1, true, 10.0);
        temporal_walk->add_multiple_edges({
            {1, 2, 100},  // Small-time differences
            {2, 3, 101},
            {3, 4, 103},
            {4, 5, 110},  // Medium time differences
            {5, 6, 130},
            {6, 7, 160},
            {7, 8, 200},  // Large time differences
            {8, 9, 250},
            {9, 10, 310}
        });
    }

    std::unique_ptr<TemporalWalk> temporal_walk;
};

// Test the constructor of TemporalWalk to ensure it initializes correctly.
TEST_F(EmptyTemporalWalkTest, ConstructorTest) {
    EXPECT_NO_THROW(temporal_walk = std::make_unique<TemporalWalk>(true));
    EXPECT_EQ(temporal_walk->get_node_count(), 0); // Assuming initial node count is 0
}


// Test adding an edge to the TemporalWalk when it's empty.
TEST_F(EmptyTemporalWalkTest, AddEdgeTest) {
    temporal_walk->add_multiple_edges({
        {1, 2, 100},
        {2, 3, 101},
        {7, 8, 102},
        {1, 7, 103},
        {3, 2, 103},
        {10, 11, 104}
    });

    EXPECT_EQ(temporal_walk->get_edge_count(), 6);
    EXPECT_EQ(temporal_walk->get_node_count(), 7);
}

// When later edges are added than the allowed max time capacity, older edges are automatically deleted.
TEST_F(EmptyTemporalWalkTestWithMaxCapacity, WhenMaxTimeCapacityExceedsEdgesAreDeletedAutomatically) {
    temporal_walk->add_multiple_edges({
        { 0, 2, 1 },
        { 2, 3, 3 },
        { 1, 9, 2 },
        { 2, 4, 3 },
        { 2, 4, 1 },
        { 1, 5, 4 }
    });

    EXPECT_EQ(temporal_walk->get_node_count(), 7);
    EXPECT_EQ(temporal_walk->get_edge_count(), 6);

    temporal_walk->add_multiple_edges({
        { 5, 6, 4 },
        { 2, 5, 4 },
        { 4, 3, 5 },
    });

    EXPECT_EQ(temporal_walk->get_node_count(), 8);
    EXPECT_EQ(temporal_walk->get_edge_count(), 9);

    temporal_walk->add_multiple_edges({
        { 1, 7, 6 }
    });

    EXPECT_EQ(temporal_walk->get_node_count(), 8);
    EXPECT_EQ(temporal_walk->get_edge_count(), 8);

    temporal_walk->add_multiple_edges({
        { 1, 5, 7 },
        { 4, 7, 8 }
    });

    EXPECT_EQ(temporal_walk->get_node_count(), 7);
    EXPECT_EQ(temporal_walk->get_edge_count(), 7);
}

// Test to check if a specific node ID is present in the filled TemporalWalk.
TEST_F(FilledDirectedTemporalWalkTest, TestNodeFoundTest) {
    const auto nodes = temporal_walk->get_node_ids();
    const auto it = std::find(nodes.begin(), nodes.end(), TEST_NODE_ID);
    EXPECT_NE(it, nodes.end());
}

// Test that the number of random walks generated matches the expected count and checks that no walk exceeds its length.
// Also test that the system can sample walks of length more than 1.
TEST_F(FilledDirectedTemporalWalkTest, WalkCountAndLensTest) {
    const auto walks = temporal_walk->get_random_walks_and_times_for_all_nodes(MAX_WALK_LEN, &linear_picker_type, 10);

    int total_walk_lens = 0;

    for (const auto& walk : walks) {
        EXPECT_LE(walk.size(), MAX_WALK_LEN) << "A walk exceeds the maximum length of " << MAX_WALK_LEN;
        EXPECT_GT(walk.size(), 0);

        total_walk_lens += static_cast<int>(walk.size());
    }

    auto average_walk_len = static_cast<float>(total_walk_lens) / static_cast<float>(walks.size());
    EXPECT_GT(average_walk_len, 1) << "System could not sample any walk of length more than 1";
}

// Test to verify that the timestamps in each walk are strictly increasing in directed graphs.
TEST_F(FilledDirectedTemporalWalkTest, WalkIncreasingTimestampTest) {
    const auto walks_forward = temporal_walk->get_random_walks_and_times_for_all_nodes(MAX_WALK_LEN, &linear_picker_type, 10);

    for (const auto& walk : walks_forward) {
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }

    const auto walks_backward = temporal_walk->get_random_walks_and_times_for_all_nodes(MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);
    for (const auto& walk : walks_backward) {
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }
}

// Test to verify that the timestamps in each walk are strictly increasing in undirected graphs.
TEST_F(FilledUndirectedTemporalWalkTest, WalkIncreasingTimestampTest) {
    const auto walks = temporal_walk->get_random_walks_and_times_for_all_nodes(MAX_WALK_LEN, &linear_picker_type, 10);

    for (const auto& walk : walks) {
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }

    const auto walks_backward = temporal_walk->get_random_walks_and_times_for_all_nodes(MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);
    for (const auto& walk : walks_backward) {
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }
}

// Test to verify that each step in walks uses valid edges from the graph
TEST_F(FilledDirectedTemporalWalkTest, WalkValidEdgesTest) {
    // Create a map of valid edges for O(1) lookup
    std::map<std::tuple<int, int, int64_t>, bool> valid_edges;
    for (const auto& edge : sample_edges) {
        valid_edges[edge] = true;
    }

    // Check forward walks
    const auto walks_forward = temporal_walk->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Forward_In_Time);

    for (const auto& walk : walks_forward) {
        if (walk.size() <= 1) continue;

        for (size_t i = 0; i < walk.size() - 1; i++) {
            int src = walk[i].node;
            int dst = walk[i+1].node;
            int64_t ts = walk[i+1].timestamp;

            bool edge_exists = valid_edges.count({src, dst, ts}) > 0;
            EXPECT_TRUE(edge_exists)
                << "Invalid forward edge in walk: (" << src << "," << dst << "," << ts << ")";
        }
    }

    // Check backward walks
    const auto walks_backward = temporal_walk->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);

    for (const auto& walk : walks_backward) {
        if (walk.size() <= 1) continue;

        for (size_t i = 1; i < walk.size(); i++) {
            int src = walk[i - 1].node;
            int dst = walk[i].node;
            int64_t ts = walk[i - 1].timestamp;

            bool edge_exists = valid_edges.count({src, dst, ts}) > 0;
            EXPECT_TRUE(edge_exists)
                << "Invalid backward edge in walk: (" << src << "," << dst << "," << ts << ")";
        }
    }
}

TEST_F(FilledDirectedTemporalWalkTest, WalkTerminalEdgesTest) {
    // For forward walks, track maximum outgoing timestamps
    std::map<int, int64_t> max_outgoing_timestamps;
    // For backward walks, track minimum incoming timestamps
    std::map<int, int64_t> min_incoming_timestamps;

    // Build timestamp maps
    for (const auto& [src, dst, ts] : sample_edges) {
        // Track max timestamp of outgoing edges for forward walks
        if (!max_outgoing_timestamps.count(src) || max_outgoing_timestamps[src] < ts) {
            max_outgoing_timestamps[src] = ts;
        }
        // Track min timestamp of incoming edges for backward walks
        if (!min_incoming_timestamps.count(dst) || min_incoming_timestamps[dst] > ts) {
            min_incoming_timestamps[dst] = ts;
        }
    }

    // Check forward walks
    const auto walks_forward = temporal_walk->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Forward_In_Time);

    for (const auto& walk : walks_forward) {
        if (walk.empty()) continue;

        // MAX_WALK_LEN approached. No need to check such walks, because they might have finished immaturely.
        if (walk.size() == MAX_WALK_LEN) continue;

        int last_node = walk.back().node;
        const int64_t last_ts = walk.back().timestamp;

        // Skip if node has no outgoing edges
        if (!max_outgoing_timestamps.count(last_node)) continue;

        int64_t max_ts = max_outgoing_timestamps[last_node];
        if (last_ts < max_ts) {
            // Check for valid edges that we could have walked to
            for (const auto& [src, dst, ts] : sample_edges) {
                if (src == last_node && ts > last_ts && ts <= max_ts) {
                    FAIL() << "Forward walk incorrectly terminated:\n"
                          << "  Node: " << last_node << "\n"
                          << "  Current timestamp: " << last_ts << "\n"
                          << "  Found valid edge at timestamp: " << ts << "\n"
                          << "  Max possible timestamp: " << max_ts << "\n"
                          << "  Edge: (" << src << "," << dst << "," << ts << ")";
                }
            }
        }
    }

    // Check backward walks
    const auto walks_backward = temporal_walk->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);

    for (const auto& walk : walks_backward) {
        if (walk.empty()) continue;

        // MAX_WALK_LEN approached. No need to check such walks, because they might have finished immaturely.
        if (walk.size() == MAX_WALK_LEN) continue;

        int first_node = walk.front().node;
        const int64_t first_ts = walk.front().timestamp;

        // Skip if node has no incoming edges
        if (!min_incoming_timestamps.count(first_node)) continue;

        int64_t min_ts = min_incoming_timestamps[first_node];
        if (first_ts > min_ts) {
            // Check for valid edges that we could have walked to
            for (const auto& [src, dst, ts] : sample_edges) {
                if (dst == first_node && ts < first_ts && ts >= min_ts) {
                    FAIL() << "Backward walk incorrectly terminated:\n"
                          << "  Node: " << first_node << "\n"
                          << "  Current timestamp: " << first_ts << "\n"
                          << "  Found valid edge at timestamp: " << ts << "\n"
                          << "  Min possible timestamp: " << min_ts << "\n"
                          << "  Edge: (" << src << "," << dst << "," << ts << ")";
                }
            }
        }
    }
}

// Test timestamps and valid edges with ExponentialWeightRandomPicker
TEST_F(FilledDirectedTemporalWalkTest, WalkIncreasingTimestampWithExponentialWeightTest) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;
    const auto walks = temporal_walk->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 10);

    for (const auto& walk : walks) {
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps not increasing at index " << i
                << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }

    const auto walks_backward = temporal_walk->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 10, nullptr, WalkDirection::Backward_In_Time);

    for (const auto& walk : walks_backward) {
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps not increasing in backward walk at index " << i
                << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }
}

TEST_F(FilledDirectedTemporalWalkTest, WalkValidEdgesWithExponentialWeightTest) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;

    // Create edge lookup map
    std::map<std::tuple<int, int, int64_t>, bool> valid_edges;
    for (const auto& edge : sample_edges) {
        valid_edges[edge] = true;
    }

    // Test forward walks
    const auto walks_forward = temporal_walk->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 10, nullptr, WalkDirection::Forward_In_Time);

    for (const auto& walk : walks_forward) {
        if (walk.size() <= 1) continue;

        for (size_t i = 0; i < walk.size() - 1; i++) {
            int src = walk[i].node;
            int dst = walk[i+1].node;
            int64_t ts = walk[i+1].timestamp;

            bool edge_exists = valid_edges.count({src, dst, ts}) > 0;
            EXPECT_TRUE(edge_exists)
                << "Invalid forward edge in exponential weight walk: ("
                << src << "," << dst << "," << ts << ")";
        }
    }

    // Test backward walks
    const auto walks_backward = temporal_walk->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 10, nullptr, WalkDirection::Backward_In_Time);

    for (const auto& walk : walks_backward) {
        if (walk.size() <= 1) continue;

        for (size_t i = 1; i < walk.size(); i++) {
            int src = walk[i - 1].node;
            int dst = walk[i].node;
            int64_t ts = walk[i - 1].timestamp;

            bool edge_exists = valid_edges.count({src, dst, ts}) > 0;
            EXPECT_TRUE(edge_exists)
                << "Invalid backward edge in exponential weight walk: ("
                << src << "," << dst << "," << ts << ")";
        }
    }
}

TEST_F(FilledDirectedTemporalWalkTest, WalkTerminalEdgesWithExponentialWeightTest) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;

    std::map<int, std::vector<int64_t>> next_valid_timestamps;
    std::map<int, std::vector<int64_t>> prev_valid_timestamps;

    for (const auto& [src, dst, ts] : sample_edges) {
        next_valid_timestamps[src].push_back(ts);
        prev_valid_timestamps[dst].push_back(ts);
    }

    for (auto& [_, timestamps] : next_valid_timestamps) {
        std::sort(timestamps.begin(), timestamps.end());
    }
    for (auto& [_, timestamps] : prev_valid_timestamps) {
        std::sort(timestamps.begin(), timestamps.end());
    }

    // Test forward walks
    const auto walks_forward = temporal_walk->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 10, nullptr, WalkDirection::Forward_In_Time);

    for (const auto& walk : walks_forward) {
        if (walk.empty() || walk.size() == MAX_WALK_LEN) continue;
        if (walk[0].timestamp == INT64_MIN) continue;  // Skip first sentinel value

        const int last_node = walk.back().node;
        const int64_t last_ts = walk.back().timestamp;

        auto it = next_valid_timestamps.find(last_node);
        if (it == next_valid_timestamps.end()) continue;

        const auto& timestamps = it->second;
        auto next_ts_it = std::upper_bound(timestamps.begin(), timestamps.end(), last_ts);

        EXPECT_EQ(next_ts_it, timestamps.end())
            << "Forward walk terminated despite having valid edges from node "
            << last_node << " after timestamp " << last_ts;
    }

    // Test backward walks
    const auto walks_backward = temporal_walk->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 10, nullptr, WalkDirection::Backward_In_Time);

    for (const auto& walk : walks_backward) {
        if (walk.empty() || walk.size() == MAX_WALK_LEN) continue;
        if (walk.back().timestamp == INT64_MAX) continue;  // Skip last sentinel value

        const int first_node = walk.front().node;
        const int64_t first_ts = walk.front().timestamp;

        auto it = prev_valid_timestamps.find(first_node);
        if (it == prev_valid_timestamps.end()) continue;

        const auto& timestamps = it->second;
        auto prev_ts_it = std::lower_bound(timestamps.begin(), timestamps.end(), first_ts);

        EXPECT_GT(prev_ts_it, timestamps.begin())
            << "Backward walk terminated despite having valid edges to node "
            << first_node << " before timestamp " << first_ts;
    }
}

TEST_F(TimescaleBoundedTemporalWalkTest, ExponentialWeightDistributionTest) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;
    constexpr int NUM_WALKS = 1000;

    std::map<std::pair<int, int>, int> edge_counts;

    const auto walks = temporal_walk->get_random_walks_and_times_for_all_nodes(
        3, &exponential_weight_picker, NUM_WALKS);

    for (const auto& walk : walks) {
        if (walk.size() < 2) continue;

        for (size_t i = 0; i < walk.size() - 1; i++) {
            edge_counts[{walk[i].node, walk[i+1].node}]++;
        }
    }

    // Check that temporal edges with smaller time differences are chosen more frequently
    for (const auto& [edge, count] : edge_counts) {
        if (const auto [src, dst] = edge; src < dst - 1) {  // Check non-consecutive node pairs
            const int next_count = edge_counts[{src, dst}];
            if (const int prev_count = edge_counts[{src, dst-1}]; prev_count > 0) {  // Only compare if both edges were traversed
                EXPECT_GE(prev_count, next_count)
                    << "Edge (" << src << "," << dst-1 << ") with smaller time difference"
                    << " was chosen less frequently than (" << src << "," << dst << ")";
            }
        }
    }
}

TEST_F(TimescaleBoundedTemporalWalkTest, WeightBasedEdgeSelection) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;
    constexpr int NUM_WALKS = 10000;

    std::map<int64_t, int> timestamp_counts;
    const auto walks = temporal_walk->get_random_walks_and_times_for_all_nodes(
        3, &exponential_weight_picker, NUM_WALKS);

    for (const auto& walk : walks) {
        if (walk.size() < 2) continue;
        for (size_t i = 1; i < walk.size(); i++) {
            timestamp_counts[walk[i].timestamp]++;
        }
    }

    // Compare consecutive timestamps (100,101,103 vs 110,130,160 vs 200,250,310)
    const std::vector<std::vector<int64_t>> timestamp_groups = {
        {100, 101, 103},      // Small differences
        {110, 130, 160},      // Medium differences
        {200, 250, 310}       // Large differences
    };

    // Within each group, closer timestamps should be selected more often
    for (const auto& group : timestamp_groups) {
        for (size_t i = 0; i < group.size() - 1; i++) {
            EXPECT_GT(timestamp_counts[group[i]], timestamp_counts[group[i + 1]])
                << "Timestamp " << group[i] << " selected less often than " << group[i + 1]
                << " despite smaller time difference";
        }
    }

    // Time difference ratios should reflect timescale bound
    for (const auto& group : timestamp_groups) {
        for (size_t i = 0; i < group.size() - 1; i++) {
            constexpr double bound = 10.0;
            if (timestamp_counts[group[i]] == 0 || timestamp_counts[group[i + 1]] == 0) continue;

            const double count_ratio = static_cast<double>(timestamp_counts[group[i + 1]]) /
                                     timestamp_counts[group[i]];
            const auto time_diff = static_cast<double>(group[i + 1] - group[i]);
            const double scaled_diff = time_diff * (bound / (310.0 - 100.0));  // Full time range
            const double expected_ratio = exp(-scaled_diff);

            EXPECT_NEAR(count_ratio, expected_ratio, 0.1)
                << "Selection ratio between timestamps " << group[i] << " and " << group[i + 1]
                << " doesn't match expected scaled exponential decay";
        }
    }
}


TEST_F(TimescaleBoundedTemporalWalkTest, ValidEdgesWithScaling) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;

    std::map<std::tuple<int, int, int64_t>, bool> valid_edges;
    for (const auto& edge : temporal_walk->get_edges()) {
        valid_edges[edge] = true;
    }

    const auto walks = temporal_walk->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 1000);

    for (const auto& walk : walks) {
        if (walk.size() <= 1) continue;

        for (size_t i = 0; i < walk.size() - 1; i++) {
            const auto edge = std::make_tuple(
                walk[i].node,
                walk[i+1].node,
                walk[i+1].timestamp
            );
            EXPECT_TRUE(valid_edges[edge])
                << "Invalid edge in timescale bounded walk: ("
                << std::get<0>(edge) << ","
                << std::get<1>(edge) << ","
                << std::get<2>(edge) << ")";
        }
    }
}

TEST_F(TimescaleBoundedTemporalWalkTest, TerminalEdgeValidation) {
    constexpr RandomPickerType exponential_weight_picker = RandomPickerType::ExponentialWeight;

    // Track valid timestamps for each node
    std::map<int, std::vector<int64_t>> next_valid_timestamps;
    for (const auto& [src, dst, ts] : temporal_walk->get_edges()) {
        next_valid_timestamps[src].push_back(ts);
    }

    // Sort timestamps
    for (auto& [_, timestamps] : next_valid_timestamps) {
        std::sort(timestamps.begin(), timestamps.end());
    }

    const auto walks = temporal_walk->get_random_walks_and_times_for_all_nodes(
        MAX_WALK_LEN, &exponential_weight_picker, 100);

    for (const auto& walk : walks) {
        if (walk.empty() || walk.size() == MAX_WALK_LEN) continue;

        const int last_node = walk.back().node;
        const int64_t last_ts = walk.back().timestamp;

        auto it = next_valid_timestamps.find(last_node);
        if (it == next_valid_timestamps.end()) continue;

        const auto& timestamps = it->second;
        auto next_ts_it = std::upper_bound(timestamps.begin(), timestamps.end(), last_ts);

        EXPECT_EQ(next_ts_it, timestamps.end())
            << "Timescale bounded walk terminated despite having valid edges from node "
            << last_node << " after timestamp " << last_ts;
    }
}
