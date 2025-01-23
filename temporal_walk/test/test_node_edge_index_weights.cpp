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
   setup_test_graph(true);

   // Helper to extract individual weights for a node's groups
   auto get_node_group_weights = [](const std::vector<double>& cumulative_weights,
                                  const std::vector<size_t>& group_offsets,
                                  size_t node) {
       std::vector<double> weights;
       size_t start = group_offsets[node];
       size_t end = group_offsets[node + 1];

       if (start < end) {
           weights.push_back(cumulative_weights[start]);
           for (size_t i = start + 1; i < end; i++) {
               weights.push_back(cumulative_weights[i] - cumulative_weights[i-1]);
           }
       }
       return weights;
   };

   // Check node 1 which has multiple timestamp groups
   int node = 1;

   // Forward weights should be higher for earlier timestamps
   auto forward_weights = get_node_group_weights(
       index.outbound_forward_weights,
       index.outbound_timestamp_group_offsets,
       node);

   for (size_t i = 0; i < forward_weights.size() - 1; i++) {
       EXPECT_GT(forward_weights[i], forward_weights[i+1])
           << "Forward weight at index " << i << " should be greater";
   }

   // Backward weights should be higher for later timestamps
   auto backward_weights = get_node_group_weights(
       index.outbound_backward_weights,
       index.outbound_timestamp_group_offsets,
       node);

   for (size_t i = 0; i < backward_weights.size() - 1; i++) {
       EXPECT_LT(backward_weights[i], backward_weights[i+1])
           << "Backward weight at index " << i << " should be smaller";
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
       size_t start = index.outbound_timestamp_group_offsets[node];
       size_t end = index.outbound_timestamp_group_offsets[node + 1];
       if (start < end) {
           EXPECT_EQ(end - start, 1);
           EXPECT_NEAR(index.outbound_forward_weights[start], 1.0, 1e-6);
           EXPECT_NEAR(index.outbound_backward_weights[start], 1.0, 1e-6);
       }
   }
}