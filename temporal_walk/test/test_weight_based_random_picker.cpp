#include <gtest/gtest.h>
#include "../src/random/WeightBasedRandomPicker.cuh"
#include "../src/utils/utils.h"

class WeightBasedRandomPickerTest : public ::testing::TestWithParam<bool>
{
protected:
    WeightBasedRandomPicker picker;

    WeightBasedRandomPickerTest(): picker(WeightBasedRandomPicker(GetParam())){}

    // Helper to verify sampling is within correct range
    void verify_sampling_range(const std::vector<double>& weights,
                               const int start,
                               const int end,
                               const int num_samples = 1000)
    {
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

TEST_P(WeightBasedRandomPickerTest, ValidationChecks)
{
    const std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};

    // Invalid start index
    EXPECT_EQ(picker.pick_random(weights, -1, 2), -1);

    // End <= start
    EXPECT_EQ(picker.pick_random(weights, 2, 2), -1);
    EXPECT_EQ(picker.pick_random(weights, 2, 1), -1);

    // End > size
    EXPECT_EQ(picker.pick_random(weights, 0, 5), -1);
}

TEST_P(WeightBasedRandomPickerTest, FullRangeSampling)
{
    verify_sampling_range({0.2, 0.5, 0.7, 1.0}, 0, 4);
}

TEST_P(WeightBasedRandomPickerTest, SubrangeSampling)
{
    // Test all subranges with the same weight vector
    verify_sampling_range({0.2, 0.5, 0.7, 1.0}, 1, 3);  // middle range
    verify_sampling_range({0.2, 0.5, 0.7, 1.0}, 0, 2);  // start range
    verify_sampling_range({0.2, 0.5, 0.7, 1.0}, 2, 4);  // end range
}

TEST_P(WeightBasedRandomPickerTest, SingleElementRange)
{
    const std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};

    // When sampling single element, should always return that index
    for (int i = 0; i < 100; i++)
    {
        EXPECT_EQ(picker.pick_random(weights, 1, 2), 1);
    }
}

TEST_P(WeightBasedRandomPickerTest, WeightDistributionTest)
{
    // Create weights with known distribution
    const std::vector<double> weights = {0.25, 0.5, 0.75, 1.0}; // Equal increments

    std::map<int, int> sample_counts;
    constexpr int num_samples = 100000;

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
        EXPECT_NEAR(proportion, 0.25, 0.01)
            << "Proportion for index " << i << " was " << proportion;
    }
}

TEST_P(WeightBasedRandomPickerTest, EdgeCaseWeights)
{
    // Test with very small weight differences
    const std::vector<double> small_diffs = {0.1, 0.100001, 0.100002, 0.100003};
    EXPECT_NE(picker.pick_random(small_diffs, 0, 4), -1);

    // Test with very large weight differences
    const std::vector<double> large_diffs = {0.1, 0.5, 0.9, 1000.0};
    EXPECT_NE(picker.pick_random(large_diffs, 0, 4), -1);
}

#ifdef HAS_CUDA
INSTANTIATE_TEST_SUITE_P(
    CPUAndGPU,
    WeightBasedRandomPickerTest,
    ::testing::Values(false, true),
    [](const testing::TestParamInfo<bool>& info) {
        return info.param ? "GPU" : "CPU";
    }
);
#else
INSTANTIATE_TEST_SUITE_P(
    CPUOnly,
    WeightBasedRandomPickerTest,
    ::testing::Values(false),
    [](const testing::TestParamInfo<bool>& info) {
        return "CPU";
    }
);
#endif
