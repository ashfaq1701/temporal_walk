#include <gtest/gtest.h>
#include <random/WeightBasedRandomPickerGPU.cuh>

#include "../src/random/WeightBasedRandomPicker.cuh"
#include "../src/common/CudaRandomStates.cuh"
#include "../src/utils/utils.h"

template<typename T>
class WeightBasedRandomPickerTest : public ::testing::Test
{
protected:
    using WeightBasedRandomPickerType = std::conditional_t<
        T::value == GPUUsageMode::ON_GPU,
        WeightBasedRandomPickerGPU<T::value>,
        WeightBasedRandomPicker<T::value>
    >;

    template <typename U>
    using VectorType = std::conditional_t<
        T::value == GPUUsageMode::ON_GPU,
        DeviceVector<U>,
        HostVector<U>
    >;

    WeightBasedRandomPickerTest() {
        if constexpr (T::value == GPUUsageMode::ON_GPU) {
            #ifdef HAS_CUDA
            CUDARandomStates::initialize();
            #endif
        }
    }

    WeightBasedRandomPickerType picker;

    // Helper to verify sampling is within correct range
    void verify_sampling_range(VectorType<double>& weights,
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
            ++sample_counts[picked];
        }

        // Verify all valid indices were sampled
        for (int i = start; i < end; i++)
        {
            EXPECT_GT(sample_counts[i], 0)
                << "Index " << i << " was never sampled";
        }
    }
};

#ifdef HAS_CUDA
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<GPUUsageMode, GPUUsageMode::ON_CPU>,
    std::integral_constant<GPUUsageMode, GPUUsageMode::ON_GPU>
>;
#else
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<GPUUsageMode, GPUUsageMode::ON_CPU>
>;
#endif

TYPED_TEST_SUITE(WeightBasedRandomPickerTest, GPU_USAGE_TYPES);

TYPED_TEST(WeightBasedRandomPickerTest, ValidationChecks)
{
    typename TestFixture::template VectorType<double> weights = {0.2, 0.5, 0.7, 1.0};

    // Invalid start index
    EXPECT_EQ(this->picker.pick_random(weights, -1, 2), -1);

    // End <= start
    EXPECT_EQ(this->picker.pick_random(weights, 2, 2), -1);
    EXPECT_EQ(this->picker.pick_random(weights, 2, 1), -1);

    // End > size
    EXPECT_EQ(this->picker.pick_random(weights, 0, 5), -1);
}

TYPED_TEST(WeightBasedRandomPickerTest, FullRangeSampling)
{
    typename TestFixture::template VectorType<double> weights = {0.2, 0.5, 0.7, 1.0};
    this->verify_sampling_range(weights, 0, 4);
}

TYPED_TEST(WeightBasedRandomPickerTest, SubrangeSampling)
{
    typename TestFixture::template VectorType<double> weights = {0.2, 0.5, 0.7, 1.0};
    // Test all subranges with the same weight vector
    this->verify_sampling_range(weights, 1, 3);  // middle range
    this->verify_sampling_range(weights, 0, 2);  // start range
    this->verify_sampling_range(weights, 2, 4);  // end range
}

TYPED_TEST(WeightBasedRandomPickerTest, SingleElementRange)
{
    typename TestFixture::template VectorType<double> weights = {0.2, 0.5, 0.7, 1.0};

    // When sampling single element, should always return that index
    for (int i = 0; i < 100; i++)
    {
        EXPECT_EQ(this->picker.pick_random(weights, 1, 2), 1);
    }
}

TYPED_TEST(WeightBasedRandomPickerTest, WeightDistributionTest)
{
    // Create weights with known distribution
    typename TestFixture::template VectorType<double> weights = {0.25, 0.5, 0.75, 1.0}; // Equal increments

    std::map<int, int> sample_counts;
    constexpr int num_samples = 100000;

    for (int i = 0; i < num_samples; i++)
    {
        int picked = this->picker.pick_random(weights, 0, 4);
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

TYPED_TEST(WeightBasedRandomPickerTest, EdgeCaseWeights)
{
    // Test with very small weight differences
    typename TestFixture::template VectorType<double> small_diffs = {0.1, 0.100001, 0.100002, 0.100003};
    EXPECT_NE(this->picker.pick_random(small_diffs, 0, 4), -1);

    // Test with very large weight differences
    typename TestFixture::template VectorType<double> large_diffs = {0.1, 0.5, 0.9, 1000.0};
    EXPECT_NE(this->picker.pick_random(large_diffs, 0, 4), -1);
}