#include <gtest/gtest.h>
#include "../src/data/TemporalGraph.h"
#include "../src/random/ExponentialWeightRandomPicker.h"
#include "../src/random/IndexBasedRandomPicker.h"

class TemporalGraphWeightTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Will be used in multiple tests
        test_edges = {
            {1, 2, 10}, // source, target, timestamp
            {1, 3, 10}, // same timestamp group
            {2, 3, 20},
            {2, 4, 20}, // same timestamp group
            {3, 4, 30},
            {4, 1, 40}
        };
    }

    // Helper to verify cumulative weights are properly normalized
    static void verify_cumulative_weights(const std::vector<double>& weights) {
        ASSERT_FALSE(weights.empty());
        for (size_t i = 0; i < weights.size(); i++) {
            EXPECT_GE(weights[i], 0.0);
            if (i > 0) {
                EXPECT_GE(weights[i], weights[i-1]);
            }
        }
        EXPECT_NEAR(weights.back(), 1.0, 1e-6);
    }

    std::vector<std::tuple<int, int, int64_t>> test_edges;
};

TEST_F(TemporalGraphWeightTest, EdgeWeightComputation) {
    TemporalGraph graph(/*directed=*/false, /*window=*/-1, /*enable_weight_computation=*/true);
    graph.add_multiple_edges(test_edges);

    // Should have 4 timestamp groups (10,20,30,40)
    const auto& fwd_weights = graph.edges.forward_cumulative_weights;
    const auto& bwd_weights = graph.edges.backward_cumulative_weights;

    ASSERT_EQ(fwd_weights.size(), 4);
    ASSERT_EQ(bwd_weights.size(), 4);

    // Verify cumulative properties
    verify_cumulative_weights(fwd_weights);
    verify_cumulative_weights(bwd_weights);

    // Forward weights should give higher probability to earlier timestamps
    for (size_t i = 0; i < fwd_weights.size() - 1; i++) {
        double curr_weight = fwd_weights[i] - (i > 0 ? fwd_weights[i-1] : 0.0);
        double next_weight = fwd_weights[i+1] - fwd_weights[i];
        EXPECT_GE(curr_weight, next_weight);
    }

    // Backward weights should give higher probability to later timestamps
    for (size_t i = 0; i < bwd_weights.size() - 1; i++) {
        double curr_weight = bwd_weights[i] - (i > 0 ? bwd_weights[i-1] : 0.0);
        double next_weight = bwd_weights[i+1] - bwd_weights[i];
        EXPECT_LE(curr_weight, next_weight);
    }
}

TEST_F(TemporalGraphWeightTest, NodeWeightComputation) {
    TemporalGraph graph(/*directed=*/true, /*window=*/-1, /*enable_weight_computation=*/true);
    graph.add_multiple_edges(test_edges);

    const auto& node_index = graph.node_index;

    // Check weights for node 2 which has both in/out edges
    int node_id = 2;
    int dense_idx = graph.node_mapping.to_dense(node_id);
    ASSERT_GE(dense_idx, 0);

    // Get node's group range
    size_t num_out_groups = node_index.get_timestamp_group_count(dense_idx, true, true);
    ASSERT_GT(num_out_groups, 0);

    // Verify weights for outbound groups
    size_t start_pos = node_index.outbound_timestamp_group_offsets[dense_idx];
    size_t end_pos = node_index.outbound_timestamp_group_offsets[dense_idx + 1];

    std::vector<double> node_out_weights(
        node_index.outbound_forward_weights.begin() + static_cast<int>(start_pos),
        node_index.outbound_forward_weights.begin() + static_cast<int>(end_pos));
    verify_cumulative_weights(node_out_weights);
}

TEST_F(TemporalGraphWeightTest, WeightBasedSampling) {
    TemporalGraph graph(/*directed=*/true, /*window=*/-1, /*enable_weight_computation=*/true);
    graph.add_multiple_edges(test_edges);

    ExponentialWeightRandomPicker picker;

    // Test forward sampling after timestamp 20
    std::map<int64_t, int> forward_samples;
    for (int i = 0; i < 1000; i++) {
        auto [src, tgt, ts] = graph.get_edge_at(picker, 20, true);
        EXPECT_GT(ts, 20);
        forward_samples[ts]++;
    }
    // Later timestamps should have lower counts
    EXPECT_GT(forward_samples[30], forward_samples[40]);

    // Test backward sampling before timestamp 30
    std::map<int64_t, int> backward_samples;
    for (int i = 0; i < 1000; i++) {
        auto [src, tgt, ts] = graph.get_edge_at(picker, 30, false);
        EXPECT_LT(ts, 30);
        backward_samples[ts]++;
    }
    // Later timestamps should have higher counts
    EXPECT_GT(backward_samples[20], backward_samples[10]);
}

TEST_F(TemporalGraphWeightTest, EdgeCases) {
    // Empty graph test
    {
        TemporalGraph empty_graph(false, -1, true);
        ExponentialWeightRandomPicker picker;

        // Initially empty - no need to call update_temporal_weights explicitly
        // as it's called in add_multiple_edges
        EXPECT_TRUE(empty_graph.edges.forward_cumulative_weights.empty());
        EXPECT_TRUE(empty_graph.edges.backward_cumulative_weights.empty());
    }

    // Single edge graph test
    {
        TemporalGraph single_edge_graph(false, -1, true);
        ExponentialWeightRandomPicker picker;

        // Add single edge (weights are updated in add_multiple_edges)
        single_edge_graph.add_multiple_edges({{1, 2, 10}});

        EXPECT_EQ(single_edge_graph.edges.forward_cumulative_weights.size(), 1);
        EXPECT_EQ(single_edge_graph.edges.backward_cumulative_weights.size(), 1);
        EXPECT_NEAR(single_edge_graph.edges.forward_cumulative_weights[0], 1.0, 1e-6);
        EXPECT_NEAR(single_edge_graph.edges.backward_cumulative_weights[0], 1.0, 1e-6);
    }
}
