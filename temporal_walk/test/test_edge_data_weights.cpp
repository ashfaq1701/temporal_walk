#include <gtest/gtest.h>
#include "../src/data/EdgeData.cuh"
#include <cmath>

class EdgeDataWeightTest : public ::testing::TestWithParam<bool> {
protected:
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

    static void add_test_edges(EdgeData& edges) {
        edges.push_back(1, 2, 10);
        edges.push_back(1, 3, 10);
        edges.push_back(2, 3, 20);
        edges.push_back(2, 4, 20);
        edges.push_back(3, 4, 30);
        edges.push_back(4, 1, 40);
        edges.update_timestamp_groups();
    }

    static std::vector<double> get_individual_weights(const VectorTypes<double>::Vector& cumulative) {
        std::vector<double> weights;
        std::visit([&weights](const auto& c) {
            weights.reserve(c.size());
            weights.push_back(c[0]);
            for (size_t i = 1; i < c.size(); i++) {
                weights.push_back(c[i] - c[i-1]);
            }
        }, cumulative);
        return weights;
    }
};

#ifdef HAS_CUDA
using USE_GPU_TYPES = ::testing::Types<std::false_type, std::true_type>;
#else
using USE_GPU_TYPES = ::testing::Types<std::false_type>;
#endif
TYPED_TEST_SUITE(EdgeDataWeightTest, USE_GPU_TYPES);

TEST_P(EdgeDataWeightTest, SingleTimestampGroup) {
   EdgeData edges(GetParam());  // CPU mode
   edges.push_back(1, 2, 10);
   edges.push_back(2, 3, 10);
   edges.update_timestamp_groups();
   edges.update_temporal_weights(-1);


    std::visit([](const auto& forward_weights, const auto& backward_weights) {
        ASSERT_EQ(forward_weights.size(), 1);
        ASSERT_EQ(backward_weights.size(), 1);
        EXPECT_NEAR(forward_weights[0], 1.0, 1e-6);
        EXPECT_NEAR(backward_weights[0], 1.0, 1e-6);
    }, edges.forward_cumulative_weights_exponential, edges.backward_cumulative_weights_exponential);
}

TEST_P(EdgeDataWeightTest, WeightNormalization) {
   EdgeData edges(GetParam());  // CPU mode
   add_test_edges(edges);
   edges.update_temporal_weights(-1);

   // Should have 4 timestamp groups (10,20,30,40)
    std::visit([](const auto& forward_weights, const auto& backward_weights)
    {
        ASSERT_EQ(forward_weights.size(), 4);
        ASSERT_EQ(backward_weights.size(), 4);
    }, edges.forward_cumulative_weights_exponential, edges.backward_cumulative_weights_exponential);

   verify_cumulative_weights(edges.forward_cumulative_weights_exponential);
   verify_cumulative_weights(edges.backward_cumulative_weights_exponential);
}

TEST_P(EdgeDataWeightTest, ForwardWeightBias) {
   EdgeData edges(GetParam());  // CPU mode
   add_test_edges(edges);
   edges.update_temporal_weights(-1);

   // Forward weights should be higher for earlier timestamps
   const std::vector<double> forward_weights = get_individual_weights(edges.forward_cumulative_weights_exponential);

   // Earlier groups should have higher weights
   for (size_t i = 0; i < forward_weights.size() - 1; i++) {
       EXPECT_GT(forward_weights[i], forward_weights[i+1])
           << "Forward weight at index " << i << " should be greater than weight at " << i+1;
   }
}

TEST_P(EdgeDataWeightTest, BackwardWeightBias) {
   EdgeData edges(GetParam());  // CPU mode
   add_test_edges(edges);
   edges.update_temporal_weights(-1);

   // Backward weights should be higher for later timestamps
   const std::vector<double> backward_weights = get_individual_weights(edges.backward_cumulative_weights_exponential);

   // Later groups should have higher weights
   for (size_t i = 0; i < backward_weights.size() - 1; i++) {
       EXPECT_LT(backward_weights[i], backward_weights[i+1])
           << "Backward weight at index " << i << " should be less than weight at " << i+1;
   }
}

TEST_P(EdgeDataWeightTest, WeightExponentialDecay) {
    EdgeData edges(GetParam());  // CPU mode
    edges.push_back(1, 2, 10);
    edges.push_back(2, 3, 20);
    edges.push_back(3, 4, 30);
    edges.update_timestamp_groups();
    edges.update_temporal_weights(-1);

    const auto forward_weights = get_individual_weights(edges.forward_cumulative_weights_exponential);
    const auto backward_weights = get_individual_weights(edges.backward_cumulative_weights_exponential);


    std::visit([&](const auto& unique_timestamps)
    {
        // For forward weights: log(w[i+1]/w[i]) = -Δt
        for (size_t i = 0; i < forward_weights.size() - 1; i++) {
            const auto time_diff = unique_timestamps[i+1] - unique_timestamps[i];
            if (forward_weights[i+1] > 0 && forward_weights[i] > 0) {
                const double log_ratio = log(forward_weights[i+1]/forward_weights[i]);
                EXPECT_NEAR(log_ratio, -time_diff, 1e-6)
                    << "Forward weight log ratio incorrect at index " << i;
            }
        }

        // For backward weights: log(w[i+1]/w[i]) = Δt
        for (size_t i = 0; i < backward_weights.size() - 1; i++) {
            const auto time_diff = unique_timestamps[i+1] - unique_timestamps[i];
            if (backward_weights[i+1] > 0 && backward_weights[i] > 0) {
                const double log_ratio = log(backward_weights[i+1]/backward_weights[i]);
                EXPECT_NEAR(log_ratio, time_diff, 1e-6)
                    << "Backward weight log ratio incorrect at index " << i;
            }
        }
    }, edges.unique_timestamps);
}

TEST_P(EdgeDataWeightTest, UpdateWeights) {
    EdgeData edges(GetParam());
    add_test_edges(edges);
    edges.update_temporal_weights(-1);

    // Store original weights
    const auto original_forward = edges.forward_cumulative_weights_exponential;
    const auto original_backward = edges.backward_cumulative_weights_exponential;

    // Add new edge with different timestamp
    edges.push_back(1, 4, 50);
    edges.update_timestamp_groups();
    edges.update_temporal_weights(-1);

    std::visit([&](const auto& original_forward_v, const auto& original_backward_v) {
        std::visit([&](const auto& forward_weights, const auto& backward_weights) {
            // Weights should be different after update
            EXPECT_NE(original_forward_v.size(), forward_weights.size());
            EXPECT_NE(original_backward_v.size(), backward_weights.size());
        }, edges.forward_cumulative_weights_exponential, edges.backward_cumulative_weights_exponential);
    }, original_forward, original_backward);

    // But should still maintain normalization
    verify_cumulative_weights(edges.forward_cumulative_weights_exponential);
    verify_cumulative_weights(edges.backward_cumulative_weights_exponential);
}

TEST_P(EdgeDataWeightTest, TimescaleBoundZero) {
    EdgeData edges(GetParam());
    add_test_edges(edges);
    edges.update_temporal_weights(0);  // Should behave like -1

    verify_cumulative_weights(edges.forward_cumulative_weights_exponential);
    verify_cumulative_weights(edges.backward_cumulative_weights_exponential);
}

TEST_P(EdgeDataWeightTest, TimescaleBoundPositive) {
    EdgeData edges(GetParam());
    add_test_edges(edges);
    constexpr double timescale_bound = 30.0;
    edges.update_temporal_weights(timescale_bound);

    std::visit([](const auto& forward_weights, const auto& backward_weights) {
        // Check relative weights instead of absolute values
        std::vector<double> forward_diffs;
        forward_diffs.push_back(forward_weights[0]);
        for (size_t i = 1; i < forward_weights.size(); i++) {
            forward_diffs.push_back(forward_weights[i] - forward_weights[i-1]);
        }

        // Earlier timestamps should have higher weights for forward
        for (size_t i = 0; i < forward_diffs.size() - 1; i++) {
            EXPECT_GT(forward_diffs[i], forward_diffs[i+1]);
        }

        // Later timestamps should have higher weights for backward
        std::vector<double> backward_diffs;
        backward_diffs.push_back(backward_weights[0]);
        for (size_t i = 1; i < backward_weights.size(); i++) {
            backward_diffs.push_back(backward_weights[i] - backward_weights[i-1]);
        }

        for (size_t i = 0; i < backward_diffs.size() - 1; i++) {
            EXPECT_LT(backward_diffs[i], backward_diffs[i+1]);
        }
    }, edges.forward_cumulative_weights_exponential, edges.backward_cumulative_weights_exponential);
}

TEST_P(EdgeDataWeightTest, ScalingComparison) {
    EdgeData edges(GetParam());
    add_test_edges(edges);

    std::vector<double> weights_unscaled, weights_scaled;

    edges.update_temporal_weights(-1);
    std::visit([&weights_unscaled](const auto& forward_weights) {
        for (size_t i = 1; i < forward_weights.size(); i++) {
            weights_unscaled.push_back(forward_weights[i] / forward_weights[i-1]);
        }
    }, edges.forward_cumulative_weights_exponential);

    edges.update_temporal_weights(50.0);
    std::visit([&weights_scaled](const auto& forward_weights) {
        for (size_t i = 1; i < forward_weights.size(); i++) {
            weights_scaled.push_back(forward_weights[i] / forward_weights[i-1]);
        }
    }, edges.forward_cumulative_weights_exponential);

    for (size_t i = 0; i < weights_unscaled.size(); i++) {
        EXPECT_NEAR(weights_scaled[i], weights_unscaled[i], 1e-2);
    }
}

TEST_P(EdgeDataWeightTest, SingleTimestampWithBounds) {
    EdgeData edges(GetParam());
    edges.push_back(1, 2, 100);
    edges.push_back(2, 3, 100);
    edges.push_back(3, 4, 100);
    edges.update_timestamp_groups();

    for (double bound : {-1.0, 0.0, 10.0, 50.0}) {
        edges.update_temporal_weights(bound);
        std::visit([](const auto& forward_weights, const auto& backward_weights) {
            ASSERT_EQ(forward_weights.size(), 1);
            ASSERT_EQ(backward_weights.size(), 1);
            EXPECT_NEAR(forward_weights[0], 1.0, 1e-6);
            EXPECT_NEAR(backward_weights[0], 1.0, 1e-6);
        }, edges.forward_cumulative_weights_exponential, edges.backward_cumulative_weights_exponential);
    }
}

TEST_P(EdgeDataWeightTest, TimescaleScalingPrecision) {
    EdgeData edges(GetParam());
    edges.push_back(1, 2, 100);
    edges.push_back(2, 3, 300);
    edges.push_back(3, 4, 700);
    edges.update_timestamp_groups();

    constexpr double timescale_bound = 2.0;
    edges.update_temporal_weights(timescale_bound);

    std::visit([](const auto& forward_weights, const auto& backward_weights, const auto& unique_ts) {
        std::vector<double> forward_diffs = get_individual_weights(forward_weights);
        std::vector<double> backward_diffs = get_individual_weights(backward_weights);

        constexpr double time_scale = timescale_bound / 600.0;

        // Check forward weights
        for (size_t i = 0; i < forward_diffs.size() - 1; i++) {
            const auto time_diff = static_cast<double>(unique_ts[i+1] - unique_ts[i]);
            const double expected_ratio = exp(-time_diff * time_scale);
            const double actual_ratio = forward_diffs[i+1] / forward_diffs[i];
            EXPECT_NEAR(actual_ratio, expected_ratio, 1e-6)
                << "Forward weight ratio incorrect at index " << i;
        }

        // Check backward weights
        for (size_t i = 0; i < backward_diffs.size() - 1; i++) {
            const auto time_diff = static_cast<double>(unique_ts[i+1] - unique_ts[i]);
            const double expected_ratio = exp(time_diff * time_scale);
            const double actual_ratio = backward_diffs[i+1] / backward_diffs[i];
            EXPECT_NEAR(actual_ratio, expected_ratio, 1e-6)
                << "Backward weight ratio incorrect at index " << i;
        }
    }, edges.forward_cumulative_weights_exponential,
       edges.backward_cumulative_weights_exponential,
       edges.unique_timestamps);
}

TEST_P(EdgeDataWeightTest, ScaledWeightBounds) {
    EdgeData edges(GetParam());
    edges.push_back(1, 2, 100);
    edges.push_back(2, 3, 300);
    edges.push_back(3, 4, 700);
    edges.update_timestamp_groups();

    constexpr double timescale_bound = 2.0;
    edges.update_temporal_weights(timescale_bound);

    const auto forward_weights = get_individual_weights(edges.forward_cumulative_weights_exponential);
    const auto backward_weights = get_individual_weights(edges.backward_cumulative_weights_exponential);

    // Maximum log ratio should not exceed timescale_bound
    for (size_t i = 0; i < forward_weights.size(); i++) {
        for (size_t j = 0; j < forward_weights.size(); j++) {
            if (forward_weights[j] > 0 && forward_weights[i] > 0) {
                const double log_ratio = log(forward_weights[i] / forward_weights[j]);
                EXPECT_LE(abs(log_ratio), timescale_bound + 1e-6)
                    << "Forward weights ratio exceeded bound at i=" << i << ", j=" << j;
            }
        }
    }

    for (size_t i = 0; i < backward_weights.size(); i++) {
        for (size_t j = 0; j < backward_weights.size(); j++) {
            if (backward_weights[j] > 0 && backward_weights[i] > 0) {
                const double log_ratio = log(backward_weights[i] / backward_weights[j]);
                EXPECT_LE(abs(log_ratio), timescale_bound + 1e-6)
                    << "Backward weights ratio exceeded bound at i=" << i << ", j=" << j;
            }
        }
    }
}

TEST_P(EdgeDataWeightTest, DifferentTimescaleBounds) {
    EdgeData edges(GetParam());
    add_test_edges(edges);

    const std::vector<double> bounds = {5.0, 10.0, 20.0};
    std::vector<std::vector<double>> scaled_ratios;

    // Collect weight ratios for different bounds
    for (const double bound : bounds) {
        edges.update_temporal_weights(bound);

        std::visit([&scaled_ratios](const auto& forward_weights) {
            std::vector<double> ratios;
            for (size_t i = 1; i < forward_weights.size(); i++) {
                ratios.push_back(forward_weights[i] / forward_weights[i-1]);
            }
            scaled_ratios.push_back(ratios);
        }, edges.forward_cumulative_weights_exponential);
    }

    // Relative ordering should be preserved across different bounds
    for (size_t i = 0; i < scaled_ratios[0].size(); i++) {
        for (size_t j = 1; j < scaled_ratios.size(); j++) {
            EXPECT_EQ(scaled_ratios[0][i] > 1.0, scaled_ratios[j][i] > 1.0)
                << "Weight ratio ordering should be consistent across different bounds";
        }
    }
}

TEST_P(EdgeDataWeightTest, WeightMonotonicity) {
    EdgeData edges(GetParam());
    add_test_edges(edges);

    const double timescale_bound = 20.0;
    edges.update_temporal_weights(timescale_bound);

    std::visit([](const auto& forward_weights, const auto& backward_weights) {
        // Forward weights should decrease monotonically
        for (size_t i = 1; i < forward_weights.size(); i++) {
            double prev_weight = i == 1 ? forward_weights[0] : forward_weights[i-1];
            double curr_weight = forward_weights[i];
            double weight_diff = curr_weight - prev_weight;

            if (i > 1) {
                double prev_diff = forward_weights[i-1] - forward_weights[i-2];
                EXPECT_GE(prev_diff, weight_diff)
                    << "Forward weight differences should decrease monotonically";
            }
        }

        // Backward weights should increase monotonically
        for (size_t i = 1; i < backward_weights.size(); i++) {
            double prev_weight = i == 1 ? backward_weights[0] : backward_weights[i-1];
            double curr_weight = backward_weights[i];
            double weight_diff = curr_weight - prev_weight;

            if (i > 1) {
                double prev_diff = backward_weights[i-1] - backward_weights[i-2];
                EXPECT_LE(prev_diff, weight_diff)
                    << "Backward weight differences should increase monotonically";
            }
        }
    }, edges.forward_cumulative_weights_exponential, edges.backward_cumulative_weights_exponential);
}

#ifdef HAS_CUDA
INSTANTIATE_TEST_SUITE_P(
    CPUAndGPU,
    EdgeDataWeightTest,
    ::testing::Values(false, true),
    [](const testing::TestParamInfo<bool>& info) {
        return info.param ? "GPU" : "CPU";
    }
);
#else
INSTANTIATE_TEST_SUITE_P(
    CPUOnly,
    EdgeDataWeightTest,
    ::testing::Values(false),
    [](const testing::TestParamInfo<bool>& info) {
        return "CPU";
    }
);
#endif
