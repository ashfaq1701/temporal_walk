#include <gtest/gtest.h>
#include "../src/data/EdgeData.h"
#include <cmath>

#include "../src/config/constants.h"

class EdgeDataWeightTest : public ::testing::Test
{
protected:
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

    static void add_test_edges(EdgeData& edges)
    {
        edges.push_back(1, 2, 10); // Group 0
        edges.push_back(1, 3, 10); // Group 0
        edges.push_back(2, 3, 20); // Group 1
        edges.push_back(2, 4, 20); // Group 1
        edges.push_back(3, 4, 30); // Group 2
        edges.push_back(4, 1, 40); // Group 3
        edges.update_timestamp_groups();
    }
};

TEST_F(EdgeDataWeightTest, EmptyEdges)
{
    EdgeData edges;
    edges.update_temporal_weights(-1);

    EXPECT_TRUE(edges.forward_weights.empty());
    EXPECT_TRUE(edges.backward_weights.empty());
}

TEST_F(EdgeDataWeightTest, SingleTimestampGroup)
{
    EdgeData edges;
    edges.push_back(1, 2, 10);
    edges.push_back(2, 3, 10);
    edges.update_timestamp_groups();
    edges.update_temporal_weights(-1);

    ASSERT_EQ(edges.forward_weights.size(), 1);
    ASSERT_EQ(edges.backward_weights.size(), 1);

    // Single group should have normalized weight of 1.0
    EXPECT_NEAR(edges.forward_weights[0], 1.0, 1e-6);
    EXPECT_NEAR(edges.backward_weights[0], 1.0, 1e-6);
}

TEST_F(EdgeDataWeightTest, WeightNormalization)
{
    EdgeData edges;
    add_test_edges(edges);
    edges.update_temporal_weights(-1);

    // Should have 4 timestamp groups (10,20,30,40)
    ASSERT_EQ(edges.forward_weights.size(), 4);
    ASSERT_EQ(edges.backward_weights.size(), 4);

    verify_weights(edges.forward_weights, true);
    verify_weights(edges.backward_weights, false);
}

TEST_F(EdgeDataWeightTest, ForwardWeightBias)
{
    EdgeData edges;
    add_test_edges(edges);
    edges.update_temporal_weights(-1);

    // Earlier groups should have higher weights
    for (size_t i = 0; i < edges.forward_weights.size() - 1; i++)
    {
        EXPECT_GT(edges.forward_weights[i], edges.forward_weights[i+1])
           << "Forward weight at index " << i << " should be greater than weight at " << i + 1;
    }
}

TEST_F(EdgeDataWeightTest, BackwardWeightBias)
{
    EdgeData edges;
    add_test_edges(edges);
    edges.update_temporal_weights(-1);

    // Later groups should have higher weights
    for (size_t i = 0; i < edges.backward_weights.size() - 1; i++)
    {
        EXPECT_LT(edges.backward_weights[i], edges.backward_weights[i+1])
           << "Backward weight at index " << i << " should be less than weight at " << i + 1;
    }
}

TEST_F(EdgeDataWeightTest, WeightExponentialDecay)
{
    EdgeData edges;
    edges.push_back(1, 2, 10);
    edges.push_back(2, 3, 20);
    edges.push_back(3, 4, 30);
    edges.update_timestamp_groups();
    edges.update_temporal_weights(-1);

    // For forward weights: log(w[i+1]/w[i]) = -Δt
    for (size_t i = 0; i < edges.forward_weights.size() - 1; i++)
    {
        const auto time_diff = edges.unique_timestamps[i + 1] - edges.unique_timestamps[i];
        if (edges.forward_weights[i + 1] > 0 && edges.forward_weights[i] > 0)
        {
            const double log_ratio = log(edges.forward_weights[i + 1] / edges.forward_weights[i]);
            EXPECT_NEAR(log_ratio, -time_diff, 1e-6)
                << "Forward weight log ratio incorrect at index " << i;
        }
    }

    // For backward weights: log(w[i+1]/w[i]) = Δt
    for (size_t i = 0; i < edges.backward_weights.size() - 1; i++)
    {
        const auto time_diff = edges.unique_timestamps[i + 1] - edges.unique_timestamps[i];
        if (edges.backward_weights[i + 1] > 0 && edges.backward_weights[i] > 0)
        {
            const double log_ratio = log(edges.backward_weights[i + 1] / edges.backward_weights[i]);
            EXPECT_NEAR(log_ratio, time_diff, 1e-6)
                << "Backward weight log ratio incorrect at index " << i;
        }
    }
}

TEST_F(EdgeDataWeightTest, UpdateWeights)
{
    EdgeData edges;
    add_test_edges(edges);
    edges.update_temporal_weights(-1);

    // Store original weights
    auto original_forward = edges.forward_weights;
    auto original_backward = edges.backward_weights;

    // Add new edge with different timestamp
    edges.push_back(1, 4, 50);
    edges.update_timestamp_groups();
    edges.update_temporal_weights(-1);

    // Weights should be different after update
    EXPECT_NE(original_forward.size(), edges.forward_weights.size());
    EXPECT_NE(original_backward.size(), edges.backward_weights.size());

    verify_weights(edges.forward_weights, true);
    verify_weights(edges.backward_weights, false);
}

TEST_F(EdgeDataWeightTest, TimescaleBoundZero)
{
    EdgeData edges;
    add_test_edges(edges);
    edges.update_temporal_weights(0); // Should behave like -1

    verify_weights(edges.forward_weights, true);
    verify_weights(edges.backward_weights, false);
}

TEST_F(EdgeDataWeightTest, TimescaleBoundPositive)
{
    EdgeData edges;
    add_test_edges(edges);
    constexpr double timescale_bound = 30.0;
    edges.update_temporal_weights(timescale_bound);

    // Earlier timestamps should have higher weights for forward
    for (size_t i = 0; i < edges.forward_weights.size() - 1; i++)
    {
        EXPECT_GT(edges.forward_weights[i], edges.forward_weights[i+1]);
    }

    // Later timestamps should have higher weights for backward
    for (size_t i = 0; i < edges.backward_weights.size() - 1; i++)
    {
        EXPECT_LT(edges.backward_weights[i], edges.backward_weights[i+1]);
    }
}

TEST_F(EdgeDataWeightTest, ScalingComparison)
{
    EdgeData edges;
    add_test_edges(edges);

    // Test relative weight proportions are preserved
    std::vector<double> weights_unscaled, weights_scaled;

    edges.update_temporal_weights(-1);
    for (size_t i = 1; i < edges.forward_weights.size(); i++)
    {
        weights_unscaled.push_back(
            edges.forward_weights[i] / edges.forward_weights[i - 1]);
    }

    edges.update_temporal_weights(50.0);
    for (size_t i = 1; i < edges.forward_weights.size(); i++)
    {
        weights_scaled.push_back(
            edges.forward_weights[i] / edges.forward_weights[i - 1]);
    }

    // Compare ratios with some tolerance
    for (size_t i = 0; i < weights_unscaled.size(); i++)
    {
        EXPECT_NEAR(weights_scaled[i], weights_unscaled[i], 1e-2);
    }
}

TEST_F(EdgeDataWeightTest, ScaledWeightBounds)
{
    EdgeData edges;
    edges.push_back(1, 2, 100);
    edges.push_back(2, 3, 300);
    edges.push_back(3, 4, 700);
    edges.update_timestamp_groups();

    constexpr double timescale_bound = 2.0;
    edges.update_temporal_weights(timescale_bound);

    // Maximum log ratio should not exceed timescale_bound
    for (size_t i = 0; i < edges.forward_weights.size(); i++)
    {
        for (size_t j = 0; j < edges.forward_weights.size(); j++)
        {
            const double log_ratio = log(edges.forward_weights[i] / edges.forward_weights[j]);
            EXPECT_LE(abs(log_ratio), timescale_bound + 1e-6);
        }
    }

    for (size_t i = 0; i < edges.backward_weights.size(); i++)
    {
        for (size_t j = 0; j < edges.backward_weights.size(); j++)
        {
            const double log_ratio = log(edges.backward_weights[i] / edges.backward_weights[j]);
            EXPECT_LE(abs(log_ratio), timescale_bound + 1e-6);
        }
    }
}

TEST_F(EdgeDataWeightTest, DifferentTimescaleBounds)
{
    EdgeData edges;
    add_test_edges(edges);

    std::vector<double> bounds = {5.0, 10.0, 20.0};
    std::vector<std::vector<double>> scaled_ratios;

    // Collect weight ratios for different bounds
    for (double bound : bounds)
    {
        edges.update_temporal_weights(bound);
        std::vector<double> ratios;
        for (size_t i = 1; i < edges.forward_weights.size(); i++)
        {
            ratios.push_back(edges.forward_weights[i] /
                edges.forward_weights[i - 1]);
        }
        scaled_ratios.push_back(ratios);
    }

    // Relative ordering should be preserved across different bounds
    for (size_t i = 0; i < scaled_ratios[0].size(); i++)
    {
        for (size_t j = 1; j < scaled_ratios.size(); j++)
        {
            EXPECT_EQ(scaled_ratios[0][i] > 1.0, scaled_ratios[j][i] > 1.0)
                << "Weight ratio ordering should be consistent across different bounds";
        }
    }
}

TEST_F(EdgeDataWeightTest, SingleTimestampWithBounds)
{
    EdgeData edges;
    // All edges have same timestamp
    edges.push_back(1, 2, 100);
    edges.push_back(2, 3, 100);
    edges.push_back(3, 4, 100);
    edges.update_timestamp_groups();

    // Test with different bounds
    for (double bound : {-1.0, 0.0, 10.0, 50.0})
    {
        edges.update_temporal_weights(bound);
        ASSERT_EQ(edges.forward_weights.size(), 1);
        ASSERT_EQ(edges.backward_weights.size(), 1);
        EXPECT_NEAR(edges.forward_weights[0], 1.0, 1e-6);
        EXPECT_NEAR(edges.backward_weights[0], 1.0, 1e-6);
    }
}

TEST_F(EdgeDataWeightTest, WeightMonotonicity)
{
    EdgeData edges;
    add_test_edges(edges);

    constexpr double timescale_bound = 20.0;
    edges.update_temporal_weights(timescale_bound);

    // Forward weights should decrease monotonically
    for (size_t i = 1; i < edges.forward_weights.size(); i++)
    {
        EXPECT_GE(edges.forward_weights[i - 1], edges.forward_weights[i])
                << "Forward weight differences should decrease monotonically";
    }

    // Backward weights should increase monotonically
    for (size_t i = 1; i < edges.backward_weights.size(); i++)
    {
        EXPECT_LE(edges.backward_weights[i - 1], edges.backward_weights[i]);
    }
}

TEST_F(EdgeDataWeightTest, TimescaleScalingPrecision)
{
    EdgeData edges;
    // Use precise timestamps for exact validation
    edges.push_back(1, 2, 100);
    edges.push_back(2, 3, 300);
    edges.push_back(3, 4, 700);
    edges.update_timestamp_groups();

    constexpr double timescale_bound = 2.0;
    edges.update_temporal_weights(timescale_bound);

    // Time range is 600, scale = 2.0/600
    constexpr double time_scale = timescale_bound / 600.0;

    // Check forward weights
    for (size_t i = 0; i < edges.forward_weights.size() - 1; i++)
    {
        const auto time_diff = static_cast<double>(
            edges.unique_timestamps[i + 1] - edges.unique_timestamps[i]);
        const double expected_ratio = exp(-time_diff * time_scale);
        const double actual_ratio = edges.forward_weights[i + 1] / edges.forward_weights[i];
        EXPECT_NEAR(actual_ratio, expected_ratio, 1e-6)
            << "Forward weight ratio incorrect at index " << i;
    }

    // Check backward weights
    for (size_t i = 0; i < edges.backward_weights.size() - 1; i++)
    {
        const auto time_diff = static_cast<double>(
            edges.unique_timestamps[i + 1] - edges.unique_timestamps[i]);
        const double expected_ratio = exp(time_diff * time_scale);
        const double actual_ratio = edges.backward_weights[i + 1] / edges.backward_weights[i];
        EXPECT_NEAR(actual_ratio, expected_ratio, 1e-6)
            << "Backward weight ratio incorrect at index " << i;
    }
}
