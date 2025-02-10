#include <gtest/gtest.h>
#include "../src/data/NodeEdgeIndex.cuh"
#include "../src/data/NodeMapping.cuh"
#include "../src/data/EdgeData.cuh"

class NodeEdgeIndexWeightTest : public ::testing::TestWithParam<bool> {
protected:
    NodeEdgeIndex index;

    NodeEdgeIndexWeightTest(): index(GetParam()) {}

    // Helper to verify weights are normalized and cumulative per node's group
    static void verify_node_weights(const VectorTypes<size_t>::Vector& group_offsets,
                                  const VectorTypes<double>::Vector& weights)
    {
        std::visit([](const auto& offs, const auto& w) {
            // For each node
            for (size_t node = 0; node < offs.size() - 1; node++) {
                size_t start = offs[node];
                size_t end = offs[node + 1];

                if (start < end) {
                    // Check weights are monotonically increasing
                    for (size_t i = start; i < end; i++) {
                        EXPECT_GE(w[i], 0.0);
                        if (i > start) {
                            EXPECT_GE(w[i], w[i-1]);
                        }
                    }
                    // Last weight in node's group should be normalized to 1.0
                    EXPECT_NEAR(w[end-1], 1.0, 1e-6);
                }
            }
        }, group_offsets, weights);
    }

    // Helper to get individual weights for a node
    static std::vector<double> get_individual_weights(
        const VectorTypes<double>::Vector& cumulative,
        const VectorTypes<size_t>::Vector& offsets,
        const size_t node) {

        std::vector<double> weights;
        std::visit([&](const auto& cum, const auto& offs) {
            const size_t start = offs[node];
            const size_t end = offs[node + 1];

            weights.push_back(cum[start]);
            for (size_t i = start + 1; i < end; i++) {
                weights.push_back(cum[i] - cum[i - 1]);
            }
        }, cumulative, offsets);

        return weights;
    }

    void setup_test_graph(bool directed = true) {
        EdgeData edges(GetParam());
        // Add edges that create multiple timestamp groups per node
        edges.push_back(1, 2, 10);
        edges.push_back(1, 3, 10); // Same timestamp group for node 1
        edges.push_back(1, 4, 20); // New timestamp group for node 1
        edges.push_back(2, 3, 20);
        edges.push_back(2, 4, 30); // Different timestamp groups for node 2
        edges.push_back(3, 4, 40);
        edges.update_timestamp_groups();

        NodeMapping mapping(GetParam());
        mapping.update(edges, 0, edges.size());

        index.rebuild(edges, mapping, directed);
        index.update_temporal_weights(edges, -1);
    }
};

TEST_P(NodeEdgeIndexWeightTest, EmptyGraph) {
    EdgeData empty_edges(GetParam());
    NodeMapping empty_mapping(GetParam());
    index.rebuild(empty_edges, empty_mapping, true);
    index.update_temporal_weights(empty_edges, -1);

    std::visit([](const auto& vec) { EXPECT_TRUE(vec.empty()); },
               index.outbound_forward_cumulative_weights_exponential);
    std::visit([](const auto& vec) { EXPECT_TRUE(vec.empty()); },
               index.outbound_backward_cumulative_weights_exponential);
    std::visit([](const auto& vec) { EXPECT_TRUE(vec.empty()); },
               index.inbound_backward_cumulative_weights_exponential);
}

TEST_P(NodeEdgeIndexWeightTest, DirectedWeightNormalization) {
    setup_test_graph(true);

    // Verify per-node weight normalization
    verify_node_weights(index.outbound_timestamp_group_offsets,
                       index.outbound_forward_cumulative_weights_exponential);
    verify_node_weights(index.outbound_timestamp_group_offsets,
                       index.outbound_backward_cumulative_weights_exponential);
    verify_node_weights(index.inbound_timestamp_group_offsets,
                       index.inbound_backward_cumulative_weights_exponential);
}

TEST_P(NodeEdgeIndexWeightTest, WeightBiasPerNode) {
    EdgeData edges(GetParam());
    edges.push_back(1, 2, 100);  // Known timestamps for precise verification
    edges.push_back(1, 3, 200);
    edges.push_back(1, 4, 300);
    edges.update_timestamp_groups();

    NodeMapping mapping(GetParam());
    mapping.update(edges, 0, edges.size());

    index.rebuild(edges, mapping, true);
    index.update_temporal_weights(edges, -1); // No scaling

    // Forward weights: exp(-(t - t_min))
    auto forward = get_individual_weights(
        index.outbound_forward_cumulative_weights_exponential,
        index.outbound_timestamp_group_offsets, 1);

    for (size_t i = 0; i < forward.size() - 1; i++) {
        double expected_ratio = exp(-100); // Time diff between groups is 100
        EXPECT_NEAR(forward[i+1]/forward[i], expected_ratio, 1e-6);
    }

    // Backward weights: exp(t - t_min)
    auto backward = get_individual_weights(
        index.outbound_backward_cumulative_weights_exponential,
        index.outbound_timestamp_group_offsets, 1);

    for (size_t i = 0; i < backward.size() - 1; i++) {
        double expected_ratio = exp(100); // Time diff between groups is 100
        EXPECT_NEAR(backward[i+1]/backward[i], expected_ratio, 1e-6);
    }
}

TEST_P(NodeEdgeIndexWeightTest, ScaledWeightRatios) {
    EdgeData edges(GetParam());
    edges.push_back(1, 2, 100);
    edges.push_back(1, 3, 300);
    edges.push_back(1, 4, 500);
    edges.update_timestamp_groups();

    NodeMapping mapping(GetParam());
    mapping.update(edges, 0, edges.size());

    index.rebuild(edges, mapping, true);

    constexpr double timescale_bound = 2.0;
    index.update_temporal_weights(edges, timescale_bound);

    const auto forward = get_individual_weights(
        index.outbound_forward_cumulative_weights_exponential,
        index.outbound_timestamp_group_offsets, 1);

    // Time range is 400, scale = 2.0/400 = 0.005
    constexpr double time_scale = timescale_bound / 400.0;

    std::visit([&](const auto& unique_ts) {
        for (size_t i = 0; i < forward.size() - 1; i++) {
            // Each step is 200 units
            double scaled_diff = 200 * time_scale;
            double expected_ratio = exp(-scaled_diff);
            EXPECT_NEAR(forward[i+1]/forward[i], expected_ratio, 1e-6)
                << "Forward ratio incorrect at index " << i;
        }

        auto backward = get_individual_weights(
            index.outbound_backward_cumulative_weights_exponential,
            index.outbound_timestamp_group_offsets, 1);

        for (size_t i = 0; i < backward.size() - 1; i++) {
            double scaled_diff = 200 * time_scale;
            double expected_ratio = exp(scaled_diff);
            EXPECT_NEAR(backward[i+1]/backward[i], expected_ratio, 1e-6)
                << "Backward ratio incorrect at index " << i;
        }
    }, edges.unique_timestamps);
}

TEST_P(NodeEdgeIndexWeightTest, UndirectedWeightNormalization) {
    setup_test_graph(false);

    // For undirected, should only have outbound weights
    verify_node_weights(index.outbound_timestamp_group_offsets,
                       index.outbound_forward_cumulative_weights_exponential);
    verify_node_weights(index.outbound_timestamp_group_offsets,
                       index.outbound_backward_cumulative_weights_exponential);

    std::visit([](const auto& vec) { EXPECT_TRUE(vec.empty()); },
               index.inbound_backward_cumulative_weights_exponential);
}

TEST_P(NodeEdgeIndexWeightTest, WeightConsistencyAcrossUpdates) {
    setup_test_graph(true);

    // Store original weights
    const auto original_out_forward = index.outbound_forward_cumulative_weights_exponential;
    const auto original_out_backward = index.outbound_backward_cumulative_weights_exponential;
    const auto original_in_backward = index.inbound_backward_cumulative_weights_exponential;

    // Rebuild and update weights again
    EdgeData edges(GetParam());
    edges.push_back(1, 2, 10);
    edges.push_back(1, 3, 10);
    edges.update_timestamp_groups();

    NodeMapping mapping(GetParam());
    mapping.update(edges, 0, edges.size());

    index.rebuild(edges, mapping, true);
    index.update_temporal_weights(edges, -1);

    // Weights should be different but still normalized
    std::visit([](const auto& orig, const auto& curr) {
        EXPECT_NE(orig.size(), curr.size());
    }, original_out_forward, index.outbound_forward_cumulative_weights_exponential);

    verify_node_weights(index.outbound_timestamp_group_offsets,
                       index.outbound_forward_cumulative_weights_exponential);
}

TEST_P(NodeEdgeIndexWeightTest, SingleTimestampGroupPerNode) {
    EdgeData edges(GetParam());
    // All edges in same timestamp group
    edges.push_back(1, 2, 10);
    edges.push_back(1, 3, 10);
    edges.push_back(2, 3, 10);
    edges.update_timestamp_groups();

    NodeMapping mapping(GetParam());
    mapping.update(edges, 0, edges.size());

    index.rebuild(edges, mapping, true);
    index.update_temporal_weights(edges, -1);

    std::visit([](const auto& offs, const auto& forward_weights, const auto& backward_weights) {
        // Each node should have single weight of 1.0
        for (size_t node = 0; node < offs.size() - 1; node++) {
            const size_t start = offs[node];
            const size_t end = offs[node + 1];
            if (start < end) {
                EXPECT_EQ(end - start, 1);
                EXPECT_NEAR(forward_weights[start], 1.0, 1e-6);
                EXPECT_NEAR(backward_weights[start], 1.0, 1e-6);
            }
        }
    }, index.outbound_timestamp_group_offsets,
       index.outbound_forward_cumulative_weights_exponential,
       index.outbound_backward_cumulative_weights_exponential);
}

TEST_P(NodeEdgeIndexWeightTest, TimescaleBoundZero) {
    EdgeData edges(GetParam());
    edges.push_back(1, 2, 10);
    edges.push_back(1, 3, 20);
    edges.push_back(1, 4, 30);
    edges.update_timestamp_groups();

    NodeMapping mapping(GetParam());
    mapping.update(edges, 0, edges.size());

    index.rebuild(edges, mapping, true);
    index.update_temporal_weights(edges, 0);  // Should behave like -1

    verify_node_weights(index.outbound_timestamp_group_offsets,
                       index.outbound_forward_cumulative_weights_exponential);
    verify_node_weights(index.outbound_timestamp_group_offsets,
                       index.outbound_backward_cumulative_weights_exponential);
}

TEST_P(NodeEdgeIndexWeightTest, TimescaleBoundWithSingleTimestamp) {
    EdgeData edges(GetParam());
    // All edges for node 1 have same timestamp
    constexpr int node_id = 1;
    edges.push_back(node_id, 2, 10);
    edges.push_back(node_id, 3, 10);
    edges.push_back(node_id, 4, 10);
    edges.update_timestamp_groups();

    NodeMapping mapping(GetParam());
    mapping.update(edges, 0, edges.size());

    const int dense_idx = mapping.to_dense(node_id);
    ASSERT_GE(dense_idx, 0) << "Node " << node_id << " not found in mapping";

    index.rebuild(edges, mapping, true);

    // Test with different bounds
    for (const double bound : {-1.0, 0.0, 10.0, 50.0}) {
        index.update_temporal_weights(edges, bound);

        std::visit([&](const auto& offs, const auto& forward_weights, const auto& backward_weights) {
            const size_t start = offs[dense_idx];
            const size_t end = offs[dense_idx + 1];
            ASSERT_EQ(end - start, 1) << "Node should have exactly one timestamp group";
            EXPECT_NEAR(forward_weights[start], 1.0, 1e-6);
            EXPECT_NEAR(backward_weights[start], 1.0, 1e-6);
        }, index.outbound_timestamp_group_offsets,
           index.outbound_forward_cumulative_weights_exponential,
           index.outbound_backward_cumulative_weights_exponential);
    }
}

TEST_P(NodeEdgeIndexWeightTest, WeightOrderPreservation) {
    EdgeData edges(GetParam());
    edges.push_back(1, 2, 10);
    edges.push_back(1, 3, 20);
    edges.push_back(1, 4, 30);
    edges.update_timestamp_groups();

    NodeMapping mapping(GetParam());
    mapping.update(edges, 0, edges.size());
    index.rebuild(edges, mapping, true);

    // Get unscaled weights
    index.update_temporal_weights(edges, -1);
    const auto unscaled_forward = index.outbound_forward_cumulative_weights_exponential;
    const auto unscaled_backward = index.outbound_backward_cumulative_weights_exponential;

    // Get scaled weights
    index.update_temporal_weights(edges, 10.0);

    std::visit([&](const auto& offs, const auto& scaled_forward, const auto& scaled_backward,
                   const auto& unscaled_forward_vec, const auto& unscaled_backward_vec) {
        // Check relative ordering is preserved for node 1
        const size_t start = offs[1];
        const size_t end = offs[2];
        for (size_t i = start + 1; i < end; i++) {
            EXPECT_EQ(unscaled_forward_vec[i] > unscaled_forward_vec[i-1],
                     scaled_forward[i] > scaled_forward[i-1]);
            EXPECT_EQ(unscaled_backward_vec[i] > unscaled_backward_vec[i-1],
                     scaled_backward[i] > scaled_backward[i-1]);
        }
    }, index.outbound_timestamp_group_offsets,
       index.outbound_forward_cumulative_weights_exponential,
       index.outbound_backward_cumulative_weights_exponential,
       unscaled_forward, unscaled_backward);
}

TEST_P(NodeEdgeIndexWeightTest, TimescaleNormalizationTest) {
    EdgeData edges(GetParam());
    // Create edges with widely varying time differences
    edges.push_back(1, 2, 100);       // Small gap
    edges.push_back(1, 3, 200);       // 100 units
    edges.push_back(1, 4, 1000);      // 800 units
    edges.push_back(1, 5, 100000);    // Large gap
    edges.update_timestamp_groups();

    NodeMapping mapping(GetParam());
    mapping.update(edges, 0, edges.size());

    index.rebuild(edges, mapping, true);

    constexpr double timescale_bound = 5.0;
    index.update_temporal_weights(edges, timescale_bound);

    std::visit([&](const auto& offs, const auto& forward_weights, const auto& timestamps) {
        const auto weights = get_individual_weights(
            index.outbound_forward_cumulative_weights_exponential,
            index.outbound_timestamp_group_offsets,
            1  // Node index
        );

        // Check that max weight difference is bounded by timescale_bound
        double max_weight_ratio = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < weights.size(); i++) {
            for (size_t j = 0; j < weights.size(); j++) {
                if (weights[j] > 0) {
                    max_weight_ratio = std::max(max_weight_ratio, log(weights[i] / weights[j]));
                }
            }
        }
        EXPECT_LE(max_weight_ratio, timescale_bound)
            << "Maximum weight ratio exceeds timescale bound";

        // Verify that relative ordering matches time differences
        for (size_t i = 0; i < weights.size() - 1; i++) {
            const double time_ratio = static_cast<double>(timestamps[i+1] - timestamps[i]) /
                              static_cast<double>(timestamps[weights.size()-1] - timestamps[0]);
            double weight_ratio = weights[i] / weights[i+1];
            EXPECT_NEAR(log(weight_ratio), timescale_bound * time_ratio, 1e-6)
                << "Weight ratio doesn't match expected scaled time difference at " << i;
        }
    }, index.outbound_timestamp_group_offsets,
       index.outbound_forward_cumulative_weights_exponential,
       edges.unique_timestamps);
}

#ifdef HAS_CUDA
INSTANTIATE_TEST_SUITE_P(
    CPUAndGPU,
    NodeEdgeIndexWeightTest,
    ::testing::Values(false, true),
    [](const testing::TestParamInfo<bool>& info) {
        return info.param ? "GPU" : "CPU";
    }
);
#else
INSTANTIATE_TEST_SUITE_P(
    CPUOnly,
    NodeEdgeIndexWeightTest,
    ::testing::Values(false),
    [](const testing::TestParamInfo<bool>& info) {
        return "CPU";
    }
);
#endif
