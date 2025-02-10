#include <gtest/gtest.h>
#include "../src/data/TemporalGraph.cuh"
#include "../src/random/WeightBasedRandomPicker.cuh"

class TemporalGraphWeightTest : public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        test_edges = {
            {1, 2, 10},
            {1, 3, 10},
            {2, 3, 20},
            {2, 4, 20},
            {3, 4, 30},
            {4, 1, 40}
        };
    }

    static void verify_cumulative_weights(const VectorTypes<double>::Vector& weights) {
        std::visit([](const auto& w) {
            ASSERT_FALSE(w.empty());
            for (size_t i = 0; i < w.size(); i++) {
                EXPECT_GE(w[i], 0.0);
                if (i > 0) {
                    EXPECT_GE(w[i], w[i-1]);
                }
            }
            EXPECT_NEAR(w.back(), 1.0, 1e-6);
        }, weights);
    }

    static std::vector<double> get_individual_weights(const VectorTypes<double>::Vector& cumulative) {
        std::vector<double> weights;
        std::visit([&weights](const auto& c) {
            weights.push_back(c[0]);
            for (size_t i = 1; i < c.size(); i++) {
                weights.push_back(c[i] - c[i-1]);
            }
        }, cumulative);
        return weights;
    }

    std::vector<std::tuple<int, int, int64_t>> test_edges;
};

TEST_P(TemporalGraphWeightTest, EdgeWeightComputation) {
    TemporalGraph graph(/*directed=*/false, GetParam(), /*window=*/-1, /*enable_weight_computation=*/true, -1);
    graph.add_multiple_edges(test_edges);

    std::visit([](const auto& forward_weights, const auto& backward_weights) {
        // Should have 4 timestamp groups (10,20,30,40)
        ASSERT_EQ(forward_weights.size(), 4);
        ASSERT_EQ(backward_weights.size(), 4);
    }, graph.edges.forward_cumulative_weights_exponential,
       graph.edges.backward_cumulative_weights_exponential);

    verify_cumulative_weights(graph.edges.forward_cumulative_weights_exponential);
    verify_cumulative_weights(graph.edges.backward_cumulative_weights_exponential);

    auto forward_weights = get_individual_weights(graph.edges.forward_cumulative_weights_exponential);
    auto backward_weights = get_individual_weights(graph.edges.backward_cumulative_weights_exponential);

    for (size_t i = 0; i < forward_weights.size() - 1; i++) {
        EXPECT_GE(forward_weights[i], forward_weights[i+1])
            << "Forward weight at " << i << " should be >= weight at " << (i+1);
    }

    for (size_t i = 0; i < backward_weights.size() - 1; i++) {
        EXPECT_LE(backward_weights[i], backward_weights[i+1])
            << "Backward weight at " << i << " should be <= weight at " << (i+1);
    }
}

TEST_P(TemporalGraphWeightTest, NodeWeightComputation) {
    TemporalGraph graph(/*directed=*/true, GetParam(), /*window=*/-1, /*enable_weight_computation=*/true);
    graph.add_multiple_edges(test_edges);

    const auto& node_index = graph.node_index;
    constexpr int node_id = 2;
    const int dense_idx = graph.node_mapping.to_dense(node_id);
    ASSERT_GE(dense_idx, 0);

    const size_t num_out_groups = node_index.get_timestamp_group_count(dense_idx, true, true);
    ASSERT_GT(num_out_groups, 0);

    std::visit([&](const auto& offsets, const auto& weights) {
        const size_t start_pos = offsets[dense_idx];
        const size_t end_pos = offsets[dense_idx + 1];

        std::vector<double> node_out_weights(
            weights.begin() + static_cast<int>(start_pos),
            weights.begin() + static_cast<int>(end_pos));

        ASSERT_FALSE(node_out_weights.empty());
        for (size_t i = 0; i < node_out_weights.size(); i++) {
            EXPECT_GE(node_out_weights[i], 0.0);
            if (i > 0) {
                EXPECT_GE(node_out_weights[i], node_out_weights[i-1]);
            }
        }
        EXPECT_NEAR(node_out_weights.back(), 1.0, 1e-6);
    }, node_index.outbound_timestamp_group_offsets,
       node_index.outbound_forward_cumulative_weights_exponential);
}

TEST_P(TemporalGraphWeightTest, WeightBasedSampling) {
    TemporalGraph graph(/*directed=*/true, GetParam(), /*window=*/-1, /*enable_weight_computation=*/true, -1);
    graph.add_multiple_edges(test_edges);

    WeightBasedRandomPicker picker(GetParam());

    // Test forward sampling after timestamp 20
    std::map<int64_t, int> forward_samples;
    for (int i = 0; i < 100; i++) {
        auto [src, tgt, ts] = graph.get_edge_at(picker, 20, true);
        EXPECT_GT(ts, 20) << "Forward sampled timestamp should be > 20";
        forward_samples[ts]++;
    }
    EXPECT_GT(forward_samples[30], forward_samples[40])
        << "Earlier timestamp 30 should be sampled more than 40";

    // Test backward sampling before timestamp 30
    std::map<int64_t, int> backward_samples;
    for (int i = 0; i < 100; i++) {
        auto [src, tgt, ts] = graph.get_edge_at(picker, 30, false);
        EXPECT_LT(ts, 30) << "Backward sampled timestamp should be < 30";
        backward_samples[ts]++;
    }
    EXPECT_GT(backward_samples[20], backward_samples[10])
        << "Later timestamp 30 should be sampled more than 10";

    backward_samples.clear();
    for (int i = 0; i < 100; i++) {
        auto [src, tgt, ts] = graph.get_edge_at(picker, 50, false);
        backward_samples[ts]++;
    }
    EXPECT_GT(backward_samples[40], backward_samples[30])
        << "Later timestamp 40 should be sampled more than 30";
}

TEST_P(TemporalGraphWeightTest, EdgeCases) {
    // Empty graph test
    {
        const TemporalGraph empty_graph(false, GetParam(), -1, true);
        std::visit([](const auto& vec) { EXPECT_TRUE(vec.empty()); },
                  empty_graph.edges.forward_cumulative_weights_exponential);
        std::visit([](const auto& vec) { EXPECT_TRUE(vec.empty()); },
                  empty_graph.edges.backward_cumulative_weights_exponential);
    }

    // Single edge graph test
    {
        TemporalGraph single_edge_graph(false, GetParam(), -1, true);
        single_edge_graph.add_multiple_edges({{1, 2, 10}});

        std::visit([](const auto& forward_weights, const auto& backward_weights) {
            EXPECT_EQ(forward_weights.size(), 1);
            EXPECT_EQ(backward_weights.size(), 1);
            EXPECT_NEAR(forward_weights[0], 1.0, 1e-6);
            EXPECT_NEAR(backward_weights[0], 1.0, 1e-6);
        }, single_edge_graph.edges.forward_cumulative_weights_exponential,
           single_edge_graph.edges.backward_cumulative_weights_exponential);
    }
}

TEST_P(TemporalGraphWeightTest, TimescaleBoundZero) {
    TemporalGraph graph(/*directed=*/true, GetParam(), /*window=*/-1, /*enable_weight_computation=*/true, 0);
    graph.add_multiple_edges(test_edges);

    verify_cumulative_weights(graph.edges.forward_cumulative_weights_exponential);
    verify_cumulative_weights(graph.edges.backward_cumulative_weights_exponential);
}

TEST_P(TemporalGraphWeightTest, TimescaleBoundSampling) {
    TemporalGraph scaled_graph(/*directed=*/true, GetParam(), /*window=*/-1, /*enable_weight_computation=*/true, 10.0);
    TemporalGraph unscaled_graph(/*directed=*/true, GetParam(), /*window=*/-1, /*enable_weight_computation=*/true, -1);

    scaled_graph.add_multiple_edges(test_edges);
    unscaled_graph.add_multiple_edges(test_edges);

    WeightBasedRandomPicker picker(GetParam());

    std::map<int64_t, int> scaled_samples, unscaled_samples;
    constexpr int num_samples = 1000;

    for (int i = 0; i < num_samples; i++) {
        auto [u1, i1, ts1] = scaled_graph.get_edge_at(picker, 20, true);
        auto [u2, i2, ts2] = unscaled_graph.get_edge_at(picker, 20, true);
        scaled_samples[ts1]++;
        unscaled_samples[ts2]++;
    }

    EXPECT_GT(scaled_samples[30], scaled_samples[40])
        << "Scaled sampling should prefer earlier timestamp";
    EXPECT_GT(unscaled_samples[30], unscaled_samples[40])
        << "Unscaled sampling should prefer earlier timestamp";
}

TEST_P(TemporalGraphWeightTest, WeightScalingPrecision) {
    TemporalGraph graph(/*directed=*/true, GetParam(), /*window=*/-1, /*enable_weight_computation=*/true, 2.0);

    graph.add_multiple_edges({
        {1, 2, 100},
        {1, 3, 200},
        {1, 4, 300}
    });

    const auto weights = get_individual_weights(graph.edges.forward_cumulative_weights_exponential);
    ASSERT_EQ(weights.size(), 3);

    constexpr double time_scale = 2.0 / 200.0;

    for (size_t i = 0; i < weights.size() - 1; i++) {
        constexpr auto time_diff = 100.0;
        const double expected_ratio = exp(-time_diff * time_scale);
        const double actual_ratio = weights[i+1] / weights[i];
        EXPECT_NEAR(actual_ratio, expected_ratio, 1e-6)
            << "Weight ratio incorrect at index " << i;
    }

    constexpr int node_id = 1;
    const int dense_idx = graph.node_mapping.to_dense(node_id);
    ASSERT_GE(dense_idx, 0);

    std::visit([&](const auto& offsets, const auto& node_weights) {
        const size_t start = offsets[dense_idx];
        const auto node_individual_weights = get_individual_weights(
            graph.node_index.outbound_forward_cumulative_weights_exponential);

        for (size_t i = 0; i < weights.size(); i++) {
            EXPECT_NEAR(node_individual_weights[start + i], weights[i], 1e-6)
                << "Node weight mismatch at index " << i;
        }
    }, graph.node_index.outbound_timestamp_group_offsets,
       graph.node_index.outbound_forward_cumulative_weights_exponential);
}

TEST_P(TemporalGraphWeightTest, SingleTimestampWithBounds) {
    const std::vector<std::tuple<int, int, int64_t>> single_ts_edges = {
        {1, 2, 100},
        {2, 3, 100},
        {3, 4, 100}
    };

    for (double bound : {-1.0, 0.0, 10.0, 50.0}) {
        TemporalGraph graph(/*directed=*/true, GetParam(), /*window=*/-1, /*enable_weight_computation=*/true, bound);
        graph.add_multiple_edges(single_ts_edges);

        std::visit([](const auto& forward_weights, const auto& backward_weights) {
            ASSERT_EQ(forward_weights.size(), 1);
            ASSERT_EQ(backward_weights.size(), 1);
            EXPECT_NEAR(forward_weights[0], 1.0, 1e-6);
            EXPECT_NEAR(backward_weights[0], 1.0, 1e-6);
        }, graph.edges.forward_cumulative_weights_exponential,
           graph.edges.backward_cumulative_weights_exponential);
    }
}

#ifdef HAS_CUDA
INSTANTIATE_TEST_SUITE_P(
    CPUAndGPU,
    TemporalGraphWeightTest,
    ::testing::Values(false, true),
    [](const testing::TestParamInfo<bool>& info) {
        return info.param ? "GPU" : "CPU";
    }
);
#else
INSTANTIATE_TEST_SUITE_P(
    CPUOnly,
    TemporalGraphWeightTest,
    ::testing::Values(false),
    [](const testing::TestParamInfo<bool>& info) {
        return "CPU";
    }
);
#endif
