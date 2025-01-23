#include <gtest/gtest.h>

#include "../src/config/constants.h"
#include "../src/data/NodeEdgeIndex.h"
#include "../src/data/NodeMapping.h"
#include "../src/data/EdgeData.h"

class NodeEdgeIndexWeightTest : public ::testing::Test {
protected:
   // Helper to verify weights are normalized and cumulative per node's group
   static void verify_node_weights(const std::vector<size_t>& group_offsets,
                          const std::vector<double>& weights) {

       // For each node
       for (size_t node = 0; node < group_offsets.size() - 1; node++) {
           size_t start = group_offsets[node];
           size_t end = group_offsets[node + 1];

           if (start < end) {
               // Check weights are monotonically increasing
               for (size_t i = start; i < end; i++) {
                   EXPECT_GE(weights[i], 0.0);
                   if (i > start) {
                       EXPECT_GE(weights[i], weights[i-1]);
                   }
               }
               // Last weight in node's group should be normalized to 1.0
               EXPECT_NEAR(weights[end-1], 1.0, 1e-6);
           }
       }
   }

   void setup_test_graph(bool directed = true) {
       EdgeData edges;
       // Add edges that create multiple timestamp groups per node
       edges.push_back(1, 2, 10);
       edges.push_back(1, 3, 10); // Same timestamp group for node 1
       edges.push_back(1, 4, 20); // New timestamp group for node 1
       edges.push_back(2, 3, 20);
       edges.push_back(2, 4, 30); // Different timestamp groups for node 2
       edges.push_back(3, 4, 40);
       edges.update_timestamp_groups();

       NodeMapping mapping;
       mapping.update(edges, 0, edges.size());

       index.rebuild(edges, mapping, directed);
       index.update_temporal_weights(edges, -1);
   }

   NodeEdgeIndex index;
};

TEST_F(NodeEdgeIndexWeightTest, EmptyGraph) {
   EdgeData empty_edges;
   NodeMapping empty_mapping;
   index.rebuild(empty_edges, empty_mapping, true);
   index.update_temporal_weights(empty_edges, -1);

   EXPECT_TRUE(index.outbound_forward_weights.empty());
   EXPECT_TRUE(index.outbound_backward_weights.empty());
   EXPECT_TRUE(index.inbound_backward_weights.empty());
}

TEST_F(NodeEdgeIndexWeightTest, DirectedWeightNormalization) {
   setup_test_graph(true);

   // Verify per-node weight normalization
   verify_node_weights(index.outbound_timestamp_group_offsets,
                      index.outbound_forward_weights);
   verify_node_weights(index.outbound_timestamp_group_offsets,
                      index.outbound_backward_weights);
   verify_node_weights(index.inbound_timestamp_group_offsets,
                      index.inbound_backward_weights);
}

TEST_F(NodeEdgeIndexWeightTest, UndirectedWeightNormalization) {
   setup_test_graph(false);

   // For undirected, should only have outbound weights
   verify_node_weights(index.outbound_timestamp_group_offsets,
                      index.outbound_forward_weights);
   verify_node_weights(index.outbound_timestamp_group_offsets,
                      index.outbound_backward_weights);
   EXPECT_TRUE(index.inbound_backward_weights.empty());
}

TEST_F(NodeEdgeIndexWeightTest, WeightBiasPerNode) {
    EdgeData edges;
    edges.push_back(1, 2, 100);  // Known timestamps for precise verification
    edges.push_back(1, 3, 200);
    edges.push_back(1, 4, 300);
    edges.update_timestamp_groups();

    NodeMapping mapping;
    mapping.update(edges, 0, edges.size());
    index.rebuild(edges, mapping, true);
    index.update_temporal_weights(edges, -1); // No scaling

    auto get_individual_weights = [](const std::vector<double>& cumulative,
                                   const std::vector<size_t>& offsets,
                                   size_t node) {
        std::vector<double> weights;
        size_t start = offsets[node];
        size_t end = offsets[node + 1];

        weights.push_back(cumulative[start]);
        for (size_t i = start + 1; i < end; i++) {
            weights.push_back(cumulative[i] - cumulative[i-1]);
        }
        return weights;
    };

    // Forward weights: exp(-(t - t_min))
    auto forward = get_individual_weights(
        index.outbound_forward_weights,
        index.outbound_timestamp_group_offsets, 1);

    for (size_t i = 0; i < forward.size() - 1; i++) {
        double expected_ratio = exp(-100); // Time diff between groups is 100
        EXPECT_NEAR(forward[i+1]/forward[i], expected_ratio, 1e-6);
    }

    // Backward weights: exp(t - t_min)
    auto backward = get_individual_weights(
        index.outbound_backward_weights,
        index.outbound_timestamp_group_offsets, 1);

    for (size_t i = 0; i < backward.size() - 1; i++) {
        double expected_ratio = exp(100); // Time diff between groups is 100
        EXPECT_NEAR(backward[i+1]/backward[i], expected_ratio, 1e-6);
    }
}

TEST_F(NodeEdgeIndexWeightTest, WeightConsistencyAcrossUpdates) {
   setup_test_graph(true);

   // Store original weights
   auto original_out_forward = index.outbound_forward_weights;
   auto original_out_backward = index.outbound_backward_weights;
   auto original_in_backward = index.inbound_backward_weights;

   // Rebuild and update weights again
   EdgeData edges;
   edges.push_back(1, 2, 10);
   edges.push_back(1, 3, 10);
   edges.update_timestamp_groups();

   NodeMapping mapping;
   mapping.update(edges, 0, edges.size());

   index.rebuild(edges, mapping, true);
   index.update_temporal_weights(edges, -1);

   // Weights should be different but still normalized
   EXPECT_NE(original_out_forward.size(), index.outbound_forward_weights.size());
   verify_node_weights(index.outbound_timestamp_group_offsets,
                      index.outbound_forward_weights);
}

TEST_F(NodeEdgeIndexWeightTest, SingleTimestampGroupPerNode) {
   EdgeData edges;
   // All edges in same timestamp group
   edges.push_back(1, 2, 10);
   edges.push_back(1, 3, 10);
   edges.push_back(2, 3, 10);
   edges.update_timestamp_groups();

   NodeMapping mapping;
   mapping.update(edges, 0, edges.size());

   index.rebuild(edges, mapping, true);
   index.update_temporal_weights(edges, -1);

   // Each node should have single weight of 1.0
   for (size_t node = 0; node < index.outbound_timestamp_group_offsets.size() - 1; node++) {
       const size_t start = index.outbound_timestamp_group_offsets[node];
       size_t end = index.outbound_timestamp_group_offsets[node + 1];
       if (start < end) {
           EXPECT_EQ(end - start, 1);
           EXPECT_NEAR(index.outbound_forward_weights[start], 1.0, 1e-6);
           EXPECT_NEAR(index.outbound_backward_weights[start], 1.0, 1e-6);
       }
   }
}

TEST_F(NodeEdgeIndexWeightTest, TimescaleBoundZero) {
    EdgeData edges;
    edges.push_back(1, 2, 10);
    edges.push_back(1, 3, 20);
    edges.push_back(1, 4, 30);
    edges.update_timestamp_groups();

    NodeMapping mapping;
    mapping.update(edges, 0, edges.size());

    index.rebuild(edges, mapping, true);
    index.update_temporal_weights(edges, 0);  // Should behave like -1

    verify_node_weights(index.outbound_timestamp_group_offsets,
                       index.outbound_forward_weights);
    verify_node_weights(index.outbound_timestamp_group_offsets,
                       index.outbound_backward_weights);
}

TEST_F(NodeEdgeIndexWeightTest, TimescaleBoundWithSingleTimestamp) {
    EdgeData edges;
    // All edges for node 1 have same timestamp
    constexpr int node_id = 1;  // Original node ID
    edges.push_back(node_id, 2, 10);
    edges.push_back(node_id, 3, 10);
    edges.push_back(node_id, 4, 10);
    edges.update_timestamp_groups();

    NodeMapping mapping;
    mapping.update(edges, 0, edges.size());

    // Get the dense index for node_id
    const int dense_idx = mapping.to_dense(node_id);
    ASSERT_GE(dense_idx, 0) << "Node " << node_id << " not found in mapping";

    index.rebuild(edges, mapping, true);

    // Test with different bounds
    for (const double bound : {-1.0, 0.0, 10.0, 50.0}) {
        index.update_temporal_weights(edges, bound);

        // Node should have single group with weight 1.0
        const size_t start = index.outbound_timestamp_group_offsets[dense_idx];
        const size_t end = index.outbound_timestamp_group_offsets[dense_idx + 1];
        ASSERT_EQ(end - start, 1) << "Node should have exactly one timestamp group";
        EXPECT_NEAR(index.outbound_forward_weights[start], 1.0, 1e-6);
        EXPECT_NEAR(index.outbound_backward_weights[start], 1.0, 1e-6);
    }
}

TEST_F(NodeEdgeIndexWeightTest, ScaledWeightRatios) {
    EdgeData edges;
    edges.push_back(1, 2, 100);
    edges.push_back(1, 3, 300);
    edges.push_back(1, 4, 500);
    edges.update_timestamp_groups();

    NodeMapping mapping;
    mapping.update(edges, 0, edges.size());
    index.rebuild(edges, mapping, true);

    constexpr double timescale_bound = 2.0;
    index.update_temporal_weights(edges, timescale_bound);

    auto get_individual_weights = [](const std::vector<double>& cumulative,
                                   const std::vector<size_t>& offsets,
                                   size_t node) {
        std::vector<double> weights;
        size_t start = offsets[node];
        size_t end = offsets[node + 1];

        weights.push_back(cumulative[start]);
        for (size_t i = start + 1; i < end; i++) {
            weights.push_back(cumulative[i] - cumulative[i-1]);
        }
        return weights;
    };

    const auto forward = get_individual_weights(
        index.outbound_forward_weights,
        index.outbound_timestamp_group_offsets, 1);

    // Time range is 400, scale = 2.0/400 = 0.005
    constexpr double time_scale = timescale_bound / 400.0;

    for (size_t i = 0; i < forward.size() - 1; i++) {
        // Each step is 200 units
        double scaled_diff = 200 * time_scale;
        double expected_ratio = exp(-scaled_diff);
        EXPECT_NEAR(forward[i+1]/forward[i], expected_ratio, 1e-6);
    }

    auto backward = get_individual_weights(
        index.outbound_backward_weights,
        index.outbound_timestamp_group_offsets, 1);

    for (size_t i = 0; i < backward.size() - 1; i++) {
        constexpr double scaled_diff = 200 * time_scale;
        const double expected_ratio = exp(scaled_diff);
        EXPECT_NEAR(backward[i+1]/backward[i], expected_ratio, 1e-6);
    }
}

TEST_F(NodeEdgeIndexWeightTest, WeightOrderPreservation) {
    EdgeData edges;
    edges.push_back(1, 2, 10);
    edges.push_back(1, 3, 20);
    edges.push_back(1, 4, 30);
    edges.update_timestamp_groups();

    NodeMapping mapping;
    mapping.update(edges, 0, edges.size());
    index.rebuild(edges, mapping, true);

    // Get unscaled weights
    index.update_temporal_weights(edges, -1);
    const auto unscaled_forward = index.outbound_forward_weights;
    const auto unscaled_backward = index.outbound_backward_weights;

    // Get scaled weights
    index.update_temporal_weights(edges, 10.0);

    // Check relative ordering is preserved for node 1
    const size_t start = index.outbound_timestamp_group_offsets[1];
    const size_t end = index.outbound_timestamp_group_offsets[2];
    for (size_t i = start + 1; i < end; i++) {
        // If unscaled weights were increasing/decreasing, scaled weights should follow
        EXPECT_EQ(unscaled_forward[i] > unscaled_forward[i-1],
                  index.outbound_forward_weights[i] > index.outbound_forward_weights[i-1]);
        EXPECT_EQ(unscaled_backward[i] > unscaled_backward[i-1],
                  index.outbound_backward_weights[i] > index.outbound_backward_weights[i-1]);
    }
}

TEST_F(NodeEdgeIndexWeightTest, TimescaleNormalizationTest) {
    EdgeData edges;
    // Create edges with widely varying time differences
    edges.push_back(1, 2, 100);       // Small gap
    edges.push_back(1, 3, 200);       // 100 units
    edges.push_back(1, 4, 1000);      // 800 units
    edges.push_back(1, 5, 100000);    // Large gap
    edges.update_timestamp_groups();

    NodeMapping mapping;
    mapping.update(edges, 0, edges.size());
    index.rebuild(edges, mapping, true);

    const double timescale_bound = 5.0;
    index.update_temporal_weights(edges, timescale_bound);

    // Helper to get individual weights from cumulative
    auto get_weights = [](const std::vector<double>& cumulative, size_t start, size_t end) {
        std::vector<double> weights;
        weights.push_back(cumulative[start]);
        for (size_t i = start + 1; i < end; i++) {
            weights.push_back(cumulative[i] - cumulative[i-1]);
        }
        return weights;
    };

    // Get forward weights for node 1
    const size_t start = index.outbound_timestamp_group_offsets[1];
    const size_t end = index.outbound_timestamp_group_offsets[2];
    const auto weights = get_weights(index.outbound_forward_weights, start, end);

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
    constexpr int64_t timestamps[] = {100, 200, 1000, 100000};
    for (size_t i = 0; i < weights.size() - 1; i++) {
        const double time_ratio = static_cast<double>(timestamps[i+1] - timestamps[i]) /
                          static_cast<double>(timestamps[weights.size()-1] - timestamps[0]);
        double weight_ratio = weights[i] / weights[i+1];
        EXPECT_NEAR(log(weight_ratio), timescale_bound * time_ratio, 1e-6)
            << "Weight ratio doesn't match expected scaled time difference at " << i;
    }
}
