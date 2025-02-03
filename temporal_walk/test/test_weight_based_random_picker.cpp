#include <gtest/gtest.h>
#include "../src/random/WeightBasedRandomPicker.cuh"
#include "../src/cuda/DualVector.cuh"

class WeightBasedRandomPickerTest : public ::testing::Test
{
protected:
    WeightBasedRandomPicker picker;

    // Helper to create a DualVector with weights
    static DualVector<double> create_weight_vector(const std::vector<double>& weights, bool use_gpu = false) {
        DualVector<double> weight_vec(use_gpu);

        if (use_gpu) {
            #ifdef HAS_CUDA
            // Use thrust::copy for GPU
            const thrust::device_vector<double> d_vec(weights.begin(), weights.end());
            weight_vec.set_device_vector(d_vec);
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        } else {
            // Direct vector assignment for CPU
            weight_vec.set_host_vector(weights);
        }

        return weight_vec;
    }

    // Helper to verify sampling is within correct range
    void verify_sampling_range(const std::vector<double>& init_weights,
                               const int start,
                               const int end,
                               const int num_samples = 1000)
    {
        auto weights = create_weight_vector(init_weights);
        std::map<int, int> sample_counts;
        for (int i = 0; i < num_samples; i++)
        {
            int picked = picker.pick_random(weights, start, end);
            EXPECT_GE(picked, start) << "Sampled index below start";
            EXPECT_LT(picked, end) << "Sampled index at or above end";
            sample_counts[picked]++;
        }

        // Verify all valid indices were sampled
        for (int i = start; i < end; i++)
        {
            EXPECT_GT(sample_counts[i], 0)
                << "Index " << i << " was never sampled";
        }
    }
};

TEST_F(WeightBasedRandomPickerTest, ValidationChecks)
{
    const auto weights = create_weight_vector({0.2, 0.5, 0.7, 1.0});

    // Invalid start index
    EXPECT_EQ(picker.pick_random(weights, -1, 2), -1);

    // End <= start
    EXPECT_EQ(picker.pick_random(weights, 2, 2), -1);
    EXPECT_EQ(picker.pick_random(weights, 2, 1), -1);

    // End > size
    EXPECT_EQ(picker.pick_random(weights, 0, 5), -1);
}

TEST_F(WeightBasedRandomPickerTest, FullRangeSampling)
{
    verify_sampling_range({0.2, 0.5, 0.7, 1.0}, 0, 4);
}

TEST_F(WeightBasedRandomPickerTest, SubrangeSampling)
{
    // Test all subranges with the same weight vector
    verify_sampling_range({0.2, 0.5, 0.7, 1.0}, 1, 3);  // middle range
    verify_sampling_range({0.2, 0.5, 0.7, 1.0}, 0, 2);  // start range
    verify_sampling_range({0.2, 0.5, 0.7, 1.0}, 2, 4);  // end range
}

TEST_F(WeightBasedRandomPickerTest, SingleElementRange)
{
    const auto weights = create_weight_vector({0.2, 0.5, 0.7, 1.0});

    // When sampling single element, should always return that index
    for (int i = 0; i < 100; i++)
    {
        EXPECT_EQ(picker.pick_random(weights, 1, 2), 1);
    }
}

TEST_F(WeightBasedRandomPickerTest, WeightDistributionTest)
{
    // Create weights with known distribution
    const auto weights = create_weight_vector({0.25, 0.5, 0.75, 1.0}); // Equal increments

    std::map<int, int> sample_counts;
    constexpr int num_samples = 10000;

    for (int i = 0; i < num_samples; i++)
    {
        int picked = picker.pick_random(weights, 0, 4);
        sample_counts[picked]++;
    }

    // Each index should be sampled roughly equally since weights
    // have equal increments
    for (int i = 0; i < 4; i++)
    {
        const double proportion = static_cast<double>(sample_counts[i]) / num_samples;
        EXPECT_NEAR(proportion, 0.25, 0.05)
            << "Proportion for index " << i << " was " << proportion;
    }
}

TEST_F(WeightBasedRandomPickerTest, EdgeCaseWeights)
{
    // Test with very small weight differences
    const auto small_diffs = create_weight_vector({0.1, 0.100001, 0.100002, 0.100003});
    EXPECT_NE(picker.pick_random(small_diffs, 0, 4), -1);

    // Test with very large weight differences
    const auto large_diffs = create_weight_vector({0.1, 0.5, 0.9, 1000.0});
    EXPECT_NE(picker.pick_random(large_diffs, 0, 4), -1);
}
