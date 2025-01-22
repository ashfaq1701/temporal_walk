#include <gtest/gtest.h>
#include "../src/random/ExponentialWeightRandomPicker.h"

class ExponentialWeightRandomPickerTest : public ::testing::Test {
protected:
    ExponentialWeightRandomPicker picker;

    // Helper to verify sampling is within correct range
    void verify_sampling_range(const std::vector<double>& weights,
                             int start,
                             int end,
                             int num_samples = 1000) {
        std::map<int, int> sample_counts;
        for (int i = 0; i < num_samples; i++) {
            int picked = picker.pick_random(weights, start, end);
            EXPECT_GE(picked, start) << "Sampled index below start";
            EXPECT_LT(picked, end) << "Sampled index at or above end";
            sample_counts[picked]++;
        }

        // Verify all valid indices were sampled
        for (int i = start; i < end; i++) {
            EXPECT_GT(sample_counts[i], 0)
                << "Index " << i << " was never sampled";
        }
    }
};

TEST_F(ExponentialWeightRandomPickerTest, ValidationChecks) {
    std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};

    // Invalid start index
    EXPECT_EQ(picker.pick_random(weights, -1, 2), -1);

    // End <= start
    EXPECT_EQ(picker.pick_random(weights, 2, 2), -1);
    EXPECT_EQ(picker.pick_random(weights, 2, 1), -1);

    // End > size
    EXPECT_EQ(picker.pick_random(weights, 0, 5), -1);
}

TEST_F(ExponentialWeightRandomPickerTest, FullRangeSampling) {
    std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};
    verify_sampling_range(weights, 0, 4);
}

TEST_F(ExponentialWeightRandomPickerTest, SubrangeSampling) {
    std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};

    // Test middle range
    verify_sampling_range(weights, 1, 3);

    // Test start range
    verify_sampling_range(weights, 0, 2);

    // Test end range
    verify_sampling_range(weights, 2, 4);
}

TEST_F(ExponentialWeightRandomPickerTest, SingleElementRange) {
    std::vector<double> weights = {0.2, 0.5, 0.7, 1.0};

    // When sampling single element, should always return that index
    for (int i = 0; i < 100; i++) {
        EXPECT_EQ(picker.pick_random(weights, 1, 2), 1);
    }
}

TEST_F(ExponentialWeightRandomPickerTest, WeightDistributionTest) {
    // Create weights with known distribution
    std::vector<double> weights = {0.25, 0.5, 0.75, 1.0};  // Equal increments

    std::map<int, int> sample_counts;
    int num_samples = 10000;

    for (int i = 0; i < num_samples; i++) {
        int picked = picker.pick_random(weights, 0, 4);
        sample_counts[picked]++;
    }

    // Each index should be sampled roughly equally since weights
    // have equal increments
    for (int i = 0; i < 4; i++) {
        double proportion = static_cast<double>(sample_counts[i]) / num_samples;
        EXPECT_NEAR(proportion, 0.25, 0.05);  // Allow 5% deviation
    }
}

TEST_F(ExponentialWeightRandomPickerTest, EdgeCaseWeights) {
    // Test with very small weight differences
    std::vector<double> small_diffs = {0.1, 0.100001, 0.100002, 0.100003};
    EXPECT_NE(picker.pick_random(small_diffs, 0, 4), -1);

    // Test with very large weight differences
    std::vector<double> large_diffs = {0.1, 0.5, 0.9, 1000.0};
    EXPECT_NE(picker.pick_random(large_diffs, 0, 4), -1);
}
