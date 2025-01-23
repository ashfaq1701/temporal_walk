#include <gtest/gtest.h>
#include "../src/data/TemporalGraph.h"
#include "../src/random/WeightBasedRandomPicker.h"
#include "../src/random/IndexBasedRandomPicker.h"

class TemporalGraphWeightTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
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

    static void verify_weights(const std::vector<double>& weights, bool should_increase)
    {
        ASSERT_FALSE(weights.empty());
        for (size_t i = 1; i < weights.size(); i++)
        {
            if (should_increase)
            {
                EXPECT_GE(weights[i - 1], weights[i]);
            }
            else
            {
                EXPECT_LE(weights[i - 1], weights[i]);
            }
        }
    }

    std::vector<std::tuple<int, int, int64_t>> test_edges;
};

TEST_F(TemporalGraphWeightTest, EdgeWeightComputation)
{
    TemporalGraph graph(/*directed=*/false, /*window=*/-1, /*enable_weight_computation=*/true, -1);
    graph.add_multiple_edges(test_edges);

    // Should have 4 timestamp groups (10,20,30,40)
    const auto& fwd_weights = graph.edges.forward_weights_exponential;
    const auto& bwd_weights = graph.edges.backward_weights_exponential;

    ASSERT_EQ(fwd_weights.size(), 4);
    ASSERT_EQ(bwd_weights.size(), 4);

    // Forward weights should give higher probability to earlier timestamps
    for (size_t i = 0; i < fwd_weights.size() - 1; i++)
    {
        double curr_weight = fwd_weights[i];
        double next_weight = fwd_weights[i + 1];
        EXPECT_GE(curr_weight, next_weight);
    }

    // Backward weights should give higher probability to later timestamps
    for (size_t i = 0; i < bwd_weights.size() - 1; i++)
    {
        double curr_weight = bwd_weights[i];
        double next_weight = bwd_weights[i + 1];
        EXPECT_LE(curr_weight, next_weight);
    }
}

TEST_F(TemporalGraphWeightTest, NodeWeightComputation)
{
    TemporalGraph graph(/*directed=*/true, /*window=*/-1, /*enable_weight_computation=*/true);
    graph.add_multiple_edges(test_edges);

    const auto& node_index = graph.node_index;

    // Check weights for node 2 which has both in/out edges
    int node_id = 2;
    int dense_idx = graph.node_mapping.to_dense(node_id);
    ASSERT_GE(dense_idx, 0);

    // Get node's group range
    const size_t num_out_groups = node_index.get_timestamp_group_count(dense_idx, true, true);
    ASSERT_GT(num_out_groups, 0);

    // Verify weights for outbound groups
    size_t start_pos = node_index.outbound_timestamp_group_offsets[dense_idx];
    size_t end_pos = node_index.outbound_timestamp_group_offsets[dense_idx + 1];

    std::vector<double> node_out_weights(
        node_index.outbound_forward_weights_exponential.begin() + static_cast<int>(start_pos),
        node_index.outbound_forward_weights_exponential.begin() + static_cast<int>(end_pos));
    verify_weights(node_out_weights, true);
}

TEST_F(TemporalGraphWeightTest, WeightBasedSampling)
{
    TemporalGraph graph(/*directed=*/true, /*window=*/-1, /*enable_weight_computation=*/true, -1);
    graph.add_multiple_edges(test_edges);

    WeightBasedRandomPicker picker;

    // Test forward sampling after timestamp 20
    std::map<int64_t, int> forward_samples;
    for (int i = 0; i < 1000; i++)
    {
        auto [src, tgt, ts] = graph.get_edge_at(picker, 20, true);
        EXPECT_GT(ts, 20);
        forward_samples[ts]++;
    }
    // Later timestamps should have lower counts
    EXPECT_GT(forward_samples[30], forward_samples[40]);

    // Test backward sampling before timestamp 30
    std::map<int64_t, int> backward_samples;
    for (int i = 0; i < 1000; i++)
    {
        auto [src, tgt, ts] = graph.get_edge_at(picker, 30, false);
        EXPECT_LT(ts, 30);
        backward_samples[ts]++;
    }
    // Later timestamps should have higher counts
    EXPECT_GT(backward_samples[20], backward_samples[10]);
}

TEST_F(TemporalGraphWeightTest, EdgeCases)
{
    // Empty graph test
    {
        TemporalGraph empty_graph(false, -1, true);
        WeightBasedRandomPicker picker;

        // Initially empty - no need to call update_temporal_weights explicitly
        // as it's called in add_multiple_edges
        EXPECT_TRUE(empty_graph.edges.forward_weights_exponential.empty());
        EXPECT_TRUE(empty_graph.edges.backward_weights_exponential.empty());
    }

    // Single edge graph test
    {
        TemporalGraph single_edge_graph(false, -1, true);
        WeightBasedRandomPicker picker;

        // Add single edge (weights are updated in add_multiple_edges)
        single_edge_graph.add_multiple_edges({{1, 2, 10}});

        EXPECT_EQ(single_edge_graph.edges.forward_weights_exponential.size(), 1);
        EXPECT_EQ(single_edge_graph.edges.backward_weights_exponential.size(), 1);
        EXPECT_NEAR(single_edge_graph.edges.forward_weights_exponential[0], 1.0, 1e-6);
        EXPECT_NEAR(single_edge_graph.edges.backward_weights_exponential[0], 1.0, 1e-6);
    }
}

TEST_F(TemporalGraphWeightTest, TimescaleBoundZero)
{
    TemporalGraph graph(/*directed=*/true, /*window=*/-1, /*enable_weight_computation=*/true, 0);
    graph.add_multiple_edges(test_edges);

    verify_weights(graph.edges.forward_weights_exponential, true);
    verify_weights(graph.edges.backward_weights_exponential, false);
}

TEST_F(TemporalGraphWeightTest, TimescaleBoundSampling)
{
    TemporalGraph scaled_graph(/*directed=*/true, /*window=*/-1, /*enable_weight_computation=*/true, 10.0);
    TemporalGraph unscaled_graph(/*directed=*/true, /*window=*/-1, /*enable_weight_computation=*/true, -1);

    scaled_graph.add_multiple_edges(test_edges);
    unscaled_graph.add_multiple_edges(test_edges);

    WeightBasedRandomPicker picker;

    // Sample edges and verify temporal bias is preserved
    std::map<int64_t, int> scaled_samples, unscaled_samples;
    constexpr int num_samples = 1000;

    // Forward sampling
    for (int i = 0; i < num_samples; i++)
    {
        auto [u1, i1, ts1] = scaled_graph.get_edge_at(picker, 20, true);
        auto [u2, i2, ts2] = unscaled_graph.get_edge_at(picker, 20, true);
        scaled_samples[ts1]++;
        unscaled_samples[ts2]++;
    }

    // Both should maintain same ordering of preferences
    EXPECT_GT(scaled_samples[30], scaled_samples[40]);
    EXPECT_GT(unscaled_samples[30], unscaled_samples[40]);
}

TEST_F(TemporalGraphWeightTest, DifferentTimescaleBounds)
{
    const std::vector<double> bounds = {2.0, 5.0, 10.};
    WeightBasedRandomPicker picker;

    for (double bound : bounds)
    {
        constexpr int num_samples = 10000;

        TemporalGraph graph(/*directed=*/true, /*window=*/-1, /*enable_weight_computation=*/true, bound);
        graph.add_multiple_edges(test_edges);

        std::map<int64_t, int> samples;
        for (int i = 0; i < num_samples; i++)
        {
            auto [u1, i1, ts] = graph.get_edge_at(picker, -1, true);
            samples[ts]++;
        }

        // Compare consecutive timestamps
        std::vector<int64_t> timestamps = {10, 20, 30, 40};
        for (size_t i = 0; i < timestamps.size() - 1; i++)
        {
            int count_curr = samples[timestamps[i]];
            int count_next = samples[timestamps[i + 1]];

            EXPECT_GT(count_curr, count_next)
                << "At bound " << bound
                << ", timestamp " << timestamps[i] << " (" << count_curr << " samples) vs "
                << timestamps[i + 1] << " (" << count_next << " samples)";
        }
    }
}

TEST_F(TemporalGraphWeightTest, SingleTimestampWithBounds)
{
    const std::vector<std::tuple<int, int, int64_t>> single_ts_edges = {
        {1, 2, 100},
        {2, 3, 100},
        {3, 4, 100}
    };

    // Test with different bounds
    for (double bound : {-1.0, 0.0, 10.0, 50.0})
    {
        TemporalGraph graph(/*directed=*/true, /*window=*/-1, /*enable_weight_computation=*/true, bound);
        graph.add_multiple_edges(single_ts_edges);

        ASSERT_EQ(graph.edges.forward_weights_exponential.size(), 1);
        ASSERT_EQ(graph.edges.backward_weights_exponential.size(), 1);
        EXPECT_NEAR(graph.edges.forward_weights_exponential[0], 1.0, 1e-6);
        EXPECT_NEAR(graph.edges.backward_weights_exponential[0], 1.0, 1e-6);
    }
}

TEST_F(TemporalGraphWeightTest, WeightScalingPrecision)
{
    TemporalGraph graph(/*directed=*/true, /*window=*/-1, /*enable_weight_computation=*/true, 2.0);

    // Use exact timestamps for precise validation
    graph.add_multiple_edges({
        {1, 2, 100},
        {1, 3, 200},
        {1, 4, 300}
    });

    const auto& edge_weights = graph.edges.forward_weights_exponential;
    ASSERT_EQ(edge_weights.size(), 3);

    // Time range is 200, scale = 2.0/200 = 0.01
    constexpr double time_scale = 2.0 / 200.0;

    for (size_t i = 0; i < edge_weights.size() - 1; i++)
    {
        constexpr auto time_diff = 100.0; // Fixed time difference
        const double expected_ratio = exp(-time_diff * time_scale);
        const double actual_ratio = edge_weights[i + 1] / edge_weights[i];
        EXPECT_NEAR(actual_ratio, expected_ratio, 1e-6)
            << "Weight ratio incorrect at index " << i;
    }

    // Check node weights
    constexpr int node_id = 1;
    const int dense_idx = graph.node_mapping.to_dense(node_id);
    ASSERT_GE(dense_idx, 0);

    const auto& node_weights = graph.node_index.outbound_forward_weights_exponential;
    const size_t start = graph.node_index.outbound_timestamp_group_offsets[dense_idx];
    const size_t end = graph.node_index.outbound_timestamp_group_offsets[dense_idx + 1];

    // Verify node weights match edge weights
    for (size_t i = 0; i < node_weights.size(); i++)
    {
        EXPECT_NEAR(node_weights[i], edge_weights[i], 1e-6);
    }
}
