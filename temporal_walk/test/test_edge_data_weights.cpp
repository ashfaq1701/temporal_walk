#include <gtest/gtest.h>
#include "../src/data/EdgeData.h"
#include <cmath>

class EdgeDataWeightTest : public ::testing::Test {
protected:
   static void verify_cumulative_weights(const std::vector<double>& weights) {
       ASSERT_FALSE(weights.empty());

       // Check weights are monotonically increasing
       for (size_t i = 0; i < weights.size(); i++) {
           EXPECT_GE(weights[i], 0.0);
           if (i > 0) {
               EXPECT_GE(weights[i], weights[i-1]);
           }
       }

       // Last weight should be normalized to 1.0
       EXPECT_NEAR(weights.back(), 1.0, 1e-6);
   }

   static void add_test_edges(EdgeData& edges) {
       edges.push_back(1, 2, 10);  // Group 0
       edges.push_back(1, 3, 10);  // Group 0
       edges.push_back(2, 3, 20);  // Group 1
       edges.push_back(2, 4, 20);  // Group 1
       edges.push_back(3, 4, 30);  // Group 2
       edges.push_back(4, 1, 40);  // Group 3
       edges.update_timestamp_groups();
   }
};

TEST_F(EdgeDataWeightTest, EmptyEdges) {
   EdgeData edges;
   edges.update_temporal_weights();

   EXPECT_TRUE(edges.forward_cumulative_weights.empty());
   EXPECT_TRUE(edges.backward_cumulative_weights.empty());
}

TEST_F(EdgeDataWeightTest, SingleTimestampGroup) {
   EdgeData edges;
   edges.push_back(1, 2, 10);
   edges.push_back(2, 3, 10);
   edges.update_timestamp_groups();
   edges.update_temporal_weights();

   ASSERT_EQ(edges.forward_cumulative_weights.size(), 1);
   ASSERT_EQ(edges.backward_cumulative_weights.size(), 1);

   // Single group should have normalized weight of 1.0
   EXPECT_NEAR(edges.forward_cumulative_weights[0], 1.0, 1e-6);
   EXPECT_NEAR(edges.backward_cumulative_weights[0], 1.0, 1e-6);
}

TEST_F(EdgeDataWeightTest, WeightNormalization) {
   EdgeData edges;
   add_test_edges(edges);
   edges.update_temporal_weights();

   // Should have 4 timestamp groups (10,20,30,40)
   ASSERT_EQ(edges.forward_cumulative_weights.size(), 4);
   ASSERT_EQ(edges.backward_cumulative_weights.size(), 4);

   verify_cumulative_weights(edges.forward_cumulative_weights);
   verify_cumulative_weights(edges.backward_cumulative_weights);
}

TEST_F(EdgeDataWeightTest, ForwardWeightBias) {
   EdgeData edges;
   add_test_edges(edges);
   edges.update_temporal_weights();

   // Forward weights should be higher for earlier timestamps
   // Calculate individual group weights from cumulative
   std::vector<double> forward_weights;
   forward_weights.push_back(edges.forward_cumulative_weights[0]);
   for (size_t i = 1; i < edges.forward_cumulative_weights.size(); i++) {
       forward_weights.push_back(
           edges.forward_cumulative_weights[i] - edges.forward_cumulative_weights[i-1]);
   }

   // Earlier groups should have higher weights
   for (size_t i = 0; i < forward_weights.size() - 1; i++) {
       EXPECT_GT(forward_weights[i], forward_weights[i+1])
           << "Forward weight at index " << i << " should be greater than weight at " << i+1;
   }
}

TEST_F(EdgeDataWeightTest, BackwardWeightBias) {
   EdgeData edges;
   add_test_edges(edges);
   edges.update_temporal_weights();

   // Backward weights should be higher for later timestamps
   // Calculate individual group weights from cumulative
   std::vector<double> backward_weights;
   backward_weights.push_back(edges.backward_cumulative_weights[0]);
   for (size_t i = 1; i < edges.backward_cumulative_weights.size(); i++) {
       backward_weights.push_back(
           edges.backward_cumulative_weights[i] - edges.backward_cumulative_weights[i-1]);
   }

   // Later groups should have higher weights
   for (size_t i = 0; i < backward_weights.size() - 1; i++) {
       EXPECT_LT(backward_weights[i], backward_weights[i+1])
           << "Backward weight at index " << i << " should be less than weight at " << i+1;
   }
}

TEST_F(EdgeDataWeightTest, WeightExponentialDecay) {
    EdgeData edges;
    add_test_edges(edges);
    edges.update_temporal_weights();

    // Extract normalized group weights
    auto get_group_weights = [](const std::vector<double>& cumulative_weights) {
        std::vector<double> group_weights;
        group_weights.push_back(cumulative_weights[0]);
        for (size_t i = 1; i < cumulative_weights.size(); i++) {
            group_weights.push_back(cumulative_weights[i] - cumulative_weights[i-1]);
        }
        return group_weights;
    };

    // Get log ratios of consecutive weights
    const auto forward_weights = get_group_weights(edges.forward_cumulative_weights);
    const auto backward_weights = get_group_weights(edges.backward_cumulative_weights);

    const double epsilon = 1e-3;

    // For forward weights, check that log ratios are proportional to time differences
    for (size_t i = 0; i < forward_weights.size() - 1; i++) {
        if (forward_weights[i] > 0 && forward_weights[i+1] > 0) {
            const auto time_diff = static_cast<double>(edges.unique_timestamps[i+1] - edges.unique_timestamps[i]);
            const double log_ratio = log(forward_weights[i] / forward_weights[i+1]);
            EXPECT_NEAR(log_ratio, time_diff, epsilon);
        }
    }

    // For backward weights, check that log ratios are proportional to negative time differences
    for (size_t i = 0; i < backward_weights.size() - 1; i++) {
        if (backward_weights[i] > 0 && backward_weights[i+1] > 0) {
            const auto time_diff = static_cast<double>(edges.unique_timestamps[i+1] - edges.unique_timestamps[i]);
            const double log_ratio = log(backward_weights[i+1] / backward_weights[i]);
            EXPECT_NEAR(log_ratio, time_diff, epsilon);
        }
    }
}

TEST_F(EdgeDataWeightTest, UpdateWeights) {
   EdgeData edges;
   add_test_edges(edges);
   edges.update_temporal_weights();

   // Store original weights
   auto original_forward = edges.forward_cumulative_weights;
   auto original_backward = edges.backward_cumulative_weights;

   // Add new edge with different timestamp
   edges.push_back(1, 4, 50);
   edges.update_timestamp_groups();
   edges.update_temporal_weights();

   // Weights should be different after update
   EXPECT_NE(original_forward.size(), edges.forward_cumulative_weights.size());
   EXPECT_NE(original_backward.size(), edges.backward_cumulative_weights.size());

   // But should still maintain normalization
   verify_cumulative_weights(edges.forward_cumulative_weights);
   verify_cumulative_weights(edges.backward_cumulative_weights);
}