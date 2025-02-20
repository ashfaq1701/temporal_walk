#include "WeightBasedRandomPicker.cuh"

#include <cuda_common/config.cuh>

#ifdef HAS_CUDA
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#endif

template<GPUUsageMode GPUUsage>
int WeightBasedRandomPicker<GPUUsage>::pick_random(
    const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
    const int group_start,
    const int group_end)
{
    // Validate inputs
    if (group_start < 0 || group_end <= group_start ||
        group_end > static_cast<int>(cumulative_weights.size())) {
        return -1;
    }

    // Get start and end sums
    double start_sum = 0.0;
    if (group_start > 0 && cumulative_weights[group_start] >= cumulative_weights[group_start - 1]) {
        start_sum = cumulative_weights[group_start - 1];
    }
    const double end_sum = cumulative_weights[group_end - 1];

    if (end_sum < start_sum) {
        return -1;
    }

    // Generate random value between [start_sum, end_sum]
    const double random_val = generate_random_value(start_sum, end_sum);
    if (GPUUsage == GPUUsageMode::ON_CPU) {
        // Find the group where random_val falls
        return static_cast<int>(std::lower_bound(
            cumulative_weights.begin() + group_start,
            cumulative_weights.begin() + group_end,
            random_val) - cumulative_weights.begin());
    } else {
        #ifdef HAS_CUDA
        // Find the group where random_val falls using thrust::lower_bound
        auto it = thrust::lower_bound(
            this->get_policy(),
            thrust::next(cumulative_weights.begin(), group_start),
            thrust::next(cumulative_weights.begin(), group_end),
            random_val);
        return static_cast<int>(thrust::distance(cumulative_weights.begin(), it));
        #else
        throw std::runtime_error("GPU support is not available, only \"ON_CPU\" version is available.");
        #endif
    }
}

template class WeightBasedRandomPicker<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class WeightBasedRandomPicker<GPUUsageMode::ON_GPU_USING_CUDA>;
template class WeightBasedRandomPicker<GPUUsageMode::ON_HOST_USING_THRUST>;
#endif
