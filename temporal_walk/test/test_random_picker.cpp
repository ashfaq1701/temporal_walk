#include <gtest/gtest.h>
#include "../src/random/ExponentialIndexRandomPicker.cuh"
#include "../src/random/WeightBasedRandomPicker.cuh"
#include "../src/random/LinearRandomPicker.cuh"

constexpr int RANDOM_START = 0;
constexpr int RANDOM_END = 10000;
constexpr int RANDOM_NUM_SAMPLES = 1000000;

class RandomPickerTest : public ::testing::TestWithParam<bool> {
protected:
    LinearRandomPicker linear_picker;
    ExponentialIndexRandomPicker exp_picker;

    RandomPickerTest() :
        linear_picker(GetParam()),
        exp_picker(GetParam()) {}

    double compute_average_picks(const bool use_exponential, const bool prioritize_end) {
        double sum = 0;
        for (int i = 0; i < RANDOM_NUM_SAMPLES; i++) {
            const int pick = use_exponential ?
                exp_picker.pick_random(RANDOM_START, RANDOM_END, prioritize_end) :
                linear_picker.pick_random(RANDOM_START, RANDOM_END, prioritize_end);
            sum += pick;
        }
        return sum / RANDOM_NUM_SAMPLES;
    }
};

// Test that prioritize_end=true gives higher average than prioritize_end=false for both pickers
TEST_P(RandomPickerTest, PrioritizeEndGivesHigherAverage) {
    // For Linear Picker
    const double linear_end_prioritized = compute_average_picks(false, true);
    const double linear_start_prioritized = compute_average_picks(false, false);
    EXPECT_GT(linear_end_prioritized, linear_start_prioritized)
        << "Linear picker with prioritize_end=true should give higher average ("
        << linear_end_prioritized << ") than prioritize_end=false ("
        << linear_start_prioritized << ")";

    // For Exponential Picker
    const double exp_end_prioritized = compute_average_picks(true, true);
    const double exp_start_prioritized = compute_average_picks(true, false);
    EXPECT_GT(exp_end_prioritized, exp_start_prioritized)
        << "Exponential picker with prioritize_end=true should give higher average ("
        << exp_end_prioritized << ") than prioritize_end=false ("
        << exp_start_prioritized << ")";
}

// Test that exponential picker is more extreme than linear picker when prioritizing end
TEST_P(RandomPickerTest, ExponentialMoreExtremeForEnd) {
    const double linear_end_prioritized = compute_average_picks(false, true);
    const double exp_end_prioritized = compute_average_picks(true, true);

    EXPECT_GT(exp_end_prioritized, linear_end_prioritized)
        << "Exponential picker with prioritize_end=true should give higher average ("
        << exp_end_prioritized << ") than Linear picker ("
        << linear_end_prioritized << ")";
}

// Test that exponential picker is more extreme than linear picker when prioritizing start
TEST_P(RandomPickerTest, ExponentialMoreExtremeForStart) {
    const double linear_start_prioritized = compute_average_picks(false, false);
    const double exp_start_prioritized = compute_average_picks(true, false);

    EXPECT_LT(exp_start_prioritized, linear_start_prioritized)
        << "Exponential picker with prioritize_end=false should give lower average ("
        << exp_start_prioritized << ") than Linear picker ("
        << linear_start_prioritized << ")";
}

// Test that output is always within bounds
TEST_P(RandomPickerTest, BoundsTest) {
    const int start = 5;
    const int end = 10;
    const int num_tests = 1000;

    for (int i = 0; i < num_tests; i++) {
        int linear_result = linear_picker.pick_random(start, end, true);
        EXPECT_GE(linear_result, start);
        EXPECT_LT(linear_result, end);

        linear_result = linear_picker.pick_random(start, end, false);
        EXPECT_GE(linear_result, start);
        EXPECT_LT(linear_result, end);

        int exp_result = exp_picker.pick_random(start, end, true);
        EXPECT_GE(exp_result, start);
        EXPECT_LT(exp_result, end);

        exp_result = exp_picker.pick_random(start, end, false);
        EXPECT_GE(exp_result, start);
        EXPECT_LT(exp_result, end);
    }
}

// Test single-element range always returns that element
TEST_P(RandomPickerTest, SingleElementRangeTest) {
    constexpr int start = 5;
    constexpr int end = 6;  // Range of size 1

    // Should always return start for both true and false prioritize_end
    EXPECT_EQ(linear_picker.pick_random(start, end, true), start);
    EXPECT_EQ(linear_picker.pick_random(start, end, false), start);
    EXPECT_EQ(exp_picker.pick_random(start, end, true), start);
    EXPECT_EQ(exp_picker.pick_random(start, end, false), start);
}

// Test probabilities more deterministically for linear random picker and two elements.
TEST_P(RandomPickerTest, TwoElementRangeDistributionTestForLinearRandomPicker) {
    const int start = 0;
    const int end = 2;
    int count_ones_end_prioritized = 0;
    int count_ones_start_prioritized = 0;
    const int num_trials = RANDOM_NUM_SAMPLES;

    // Run trials
    for (int i = 0; i < num_trials; i++) {
        // Test prioritize_end=true
        int result_end = linear_picker.pick_random(start, end, true);
        if (result_end == 1) {
            count_ones_end_prioritized++;
        }

        // Test prioritize_end=false (separate trial)
        int result_start = linear_picker.pick_random(start, end, false);
        if (result_start == 1) {
            count_ones_start_prioritized++;
        }
    }

    constexpr double expected_prob_end = 2.0 / 3.0;    // probability of getting 1 when prioritizing end
    constexpr double expected_prob_start = 1.0 / 3.0;  // probability of getting 1 when prioritizing start

    const double actual_prob_end = static_cast<double>(count_ones_end_prioritized) / num_trials;
    const double actual_prob_start = static_cast<double>(count_ones_start_prioritized) / num_trials;

    constexpr double tolerance = 0.02;  // 2% tolerance

    EXPECT_NEAR(actual_prob_end, expected_prob_end, tolerance)
        << "When prioritizing end, probability of picking 1 should be approximately "
        << expected_prob_end << " but got " << actual_prob_end;

    EXPECT_NEAR(actual_prob_start, expected_prob_start, tolerance)
        << "When prioritizing start, probability of picking 1 should be approximately "
        << expected_prob_start << " but got " << actual_prob_start;
}

// Test probabilities more deterministically for exponential random picker and two elements.
TEST_P(RandomPickerTest, TwoElementRangeDistributionTestForExponentialRandomPicker) {
    const int start = 0;
    const int end = 2;
    int count_ones_end_prioritized = 0;
    int count_ones_start_prioritized = 0;
    constexpr int num_trials = RANDOM_NUM_SAMPLES;

    // Run trials
    for (int i = 0; i < num_trials; i++) {
        // Test prioritize_end=true
        int result_end = exp_picker.pick_random(start, end, true);
        if (result_end == 1) {
            count_ones_end_prioritized++;
        }

        // Test prioritize_end=false (separate trial)
        int result_start = exp_picker.pick_random(start, end, false);
        if (result_start == 1) {
            count_ones_start_prioritized++;
        }
    }

    const double e = std::exp(1.0);
    const double e_squared = e * e;
    const double expected_prob_end = (e - 1.0) * e / (e_squared - 1.0);
    const double expected_prob_start = (e - 1.0) / (e_squared - 1.0);

    const double actual_prob_end = static_cast<double>(count_ones_end_prioritized) / num_trials;
    const double actual_prob_start = static_cast<double>(count_ones_start_prioritized) / num_trials;

    constexpr double tolerance = 0.005;  // 0.5% tolerance

    EXPECT_NEAR(actual_prob_end, expected_prob_end, tolerance)
        << "When prioritizing end, probability of picking 1 should be approximately "
        << expected_prob_end << " but got " << actual_prob_end;

    EXPECT_NEAR(actual_prob_start, expected_prob_start, tolerance)
        << "When prioritizing start, probability of picking 1 should be approximately "
        << expected_prob_start << " but got " << actual_prob_start;
}

#ifdef HAS_CUDA
INSTANTIATE_TEST_SUITE_P(
    CPUAndGPU,
    RandomPickerTest,
    ::testing::Values(false, true),
    [](const testing::TestParamInfo<bool>& info) {
        return info.param ? "GPU" : "CPU";
    }
);
#else
INSTANTIATE_TEST_SUITE_P(
    CPUOnly,
    RandomPickerTest,
    ::testing::Values(false),
    [](const testing::TestParamInfo<bool>& info) {
        return "CPU";
    }
);
#endif
